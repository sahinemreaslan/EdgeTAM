"""
Basit Object Tracking Örneği - EdgeTAM
======================================

Bu örnek, x,y,w,h koordinatları ile tek bir nesneyi takip eder.

Kullanım:
    python simple_object_tracking.py --video examples/01_dog.mp4 --x 100 --y 100 --w 200 --h 200

Parametreler:
    --video: Video dosyası yolu
    --x: Bounding box x koordinatı (sol üst köşe)
    --y: Bounding box y koordinatı (sol üst köşe)
    --w: Bounding box genişliği
    --h: Bounding box yüksekliği
    --output: Çıktı video dosyası (varsayılan: output_tracking.mp4)
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path

from sam2.build_sam import build_sam2_video_predictor


def extract_frames(video_path, output_dir="temp_frames"):
    """Video'dan frame'leri çıkarır."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_names = []

    print(f"Video'dan frame'ler çıkarılıyor...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"{frame_idx:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        frame_names.append(frame_name)
        frame_idx += 1

    cap.release()
    print(f"✓ {frame_idx} frame çıkarıldı")

    return frame_names, output_dir


def apply_mask(image, mask, color=(30, 144, 255), alpha=0.5):
    """Maskeyi görüntü üzerine uygular."""
    # Renkli maske oluştur
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask > 0] = color

    # Yarı saydam birleştir
    result = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    # Kontur çiz
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color, 2)

    return result


def track_object(video_path, x, y, w, h, output_path="output_tracking.mp4"):
    """
    Video'da tek bir nesneyi takip eder.

    Args:
        video_path: Video dosyası yolu
        x, y, w, h: Nesnenin bounding box koordinatları
        output_path: Çıktı video dosyası
    """

    print("\n" + "="*60)
    print("EdgeTAM - Basit Object Tracking")
    print("="*60)

    # Cihaz seçimi
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n📱 Cihaz: {device}")

    # Autocast ayarla
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Model yükle
    print("\n🔧 Model yükleniyor...")
    checkpoint = "checkpoints/edgetam.pt"
    model_cfg = "configs/edgetam.yaml"

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    print("✓ Model yüklendi")

    # Frame'leri çıkar
    print("\n📹 Video işleniyor...")
    frame_names, frames_dir = extract_frames(video_path)

    # Inference state başlat
    print("\n🎯 Nesne takibi başlatılıyor...")
    inference_state = predictor.init_state(video_path=frames_dir)

    # Bounding box'ı oluştur (x, y, w, h -> x_min, y_min, x_max, y_max)
    box = np.array([x, y, x + w, y + h], dtype=np.float32)
    print(f"\n📦 Bounding Box: x={x}, y={y}, w={w}, h={h}")
    print(f"   Koordinatlar: [{x}, {y}, {x+w}, {y+h}]")

    # İlk frame'de nesneyi seç
    ann_frame_idx = 0  # İlk frame
    obj_id = 1  # Nesne ID'si

    with torch.inference_mode():
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            box=box,
        )

    print(f"✓ Frame {ann_frame_idx} üzerinde nesne seçildi")

    # Video boyunca takip et
    print("\n🎬 Video boyunca takip ediliyor...")
    video_segments = {}

    with torch.inference_mode():
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    print(f"✓ {len(video_segments)} frame işlendi")

    # Sonuçları video olarak kaydet
    print(f"\n💾 Sonuçlar kaydediliyor: {output_path}")

    # İlk frame'i oku
    first_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    height, width = first_frame.shape[:2]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Her frame'i işle ve kaydet
    for frame_idx, frame_name in enumerate(frame_names):
        # Frame'i oku
        frame = cv2.imread(os.path.join(frames_dir, frame_name))

        # İlk frame'de bounding box çiz
        if frame_idx == 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Initial Box", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Maskeyi ekle
        if frame_idx in video_segments and obj_id in video_segments[frame_idx]:
            mask = video_segments[frame_idx][obj_id]
            frame = apply_mask(frame, mask)

        # Frame numarasını ekle
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()

    # Geçici dosyaları temizle
    import shutil
    shutil.rmtree(frames_dir)

    print(f"✓ Video kaydedildi: {output_path}")
    print("\n" + "="*60)
    print("✅ Takip tamamlandı!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='EdgeTAM ile basit object tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
  python simple_object_tracking.py --video examples/01_dog.mp4 --x 100 --y 100 --w 200 --h 200
  python simple_object_tracking.py --video examples/01_dog.mp4 --x 50 --y 50 --w 150 --h 150 --output my_tracking.mp4
        """
    )

    parser.add_argument('--video', type=str, required=True,
                       help='Video dosyası yolu')
    parser.add_argument('--x', type=int, required=True,
                       help='Bounding box x koordinatı (sol üst köşe)')
    parser.add_argument('--y', type=int, required=True,
                       help='Bounding box y koordinatı (sol üst köşe)')
    parser.add_argument('--w', type=int, required=True,
                       help='Bounding box genişliği')
    parser.add_argument('--h', type=int, required=True,
                       help='Bounding box yüksekliği')
    parser.add_argument('--output', type=str, default='output_tracking.mp4',
                       help='Çıktı video dosyası (varsayılan: output_tracking.mp4)')

    args = parser.parse_args()

    # Video dosyasının varlığını kontrol et
    if not os.path.exists(args.video):
        print(f"❌ Hata: Video dosyası bulunamadı: {args.video}")
        return

    # Koordinatların geçerliliğini kontrol et
    if args.w <= 0 or args.h <= 0:
        print(f"❌ Hata: Genişlik ve yükseklik pozitif olmalıdır")
        return

    if args.x < 0 or args.y < 0:
        print(f"⚠️  Uyarı: x ve y koordinatları negatif")

    # Takip işlemini başlat
    track_object(
        video_path=args.video,
        x=args.x,
        y=args.y,
        w=args.w,
        h=args.h,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
