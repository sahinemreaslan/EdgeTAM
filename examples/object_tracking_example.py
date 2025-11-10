"""
EdgeTAM Object Tracking Örneği
==============================

Bu örnek, EdgeTAM kullanarak videodaki nesneleri nasıl takip edeceğinizi gösterir.
Örnek, aşağıdaki işlemleri gerçekleştirir:
1. Video'dan frame'leri okuma
2. EdgeTAM modelini yükleme
3. Nesne üzerinde tıklama veya kutu (box) ile seçim
4. Video boyunca nesneyi takip etme
5. Sonuçları görselleştirme ve kaydetme

Kullanım:
    python object_tracking_example.py --video_path <video_yolu> --output_path <cikti_yolu>
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

# EdgeTAM import'ları
from sam2.build_sam import build_sam2_video_predictor


def extract_frames_from_video(video_path, output_dir):
    """Video'dan frame'leri çıkarır ve kaydeder."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_names = []

    print(f"Video'dan frame'ler çıkarılıyor: {video_path}")

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
    print(f"Toplam {frame_idx} frame çıkarıldı.")

    return frame_names


def show_mask_on_image(image, mask, obj_id=None):
    """Maskeyi görüntü üzerine ekler."""
    # Renk paleti oluştur
    if obj_id is not None:
        cmap = plt.get_cmap("tab10")
        color = np.array([*cmap(obj_id)[:3]]) * 255
    else:
        color = np.array([30, 144, 255])  # Mavi

    # Maskeyi RGB görüntüye dönüştür
    mask_image = np.zeros_like(image, dtype=np.uint8)
    mask_image[mask > 0] = color

    # Yarı saydam olarak birleştir
    alpha = 0.5
    result = cv2.addWeighted(image, 1, mask_image, alpha, 0)

    # Kontur çiz
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, color.tolist(), 2)

    return result


def visualize_points(image, points, labels):
    """Noktaları görüntü üzerine çizer."""
    result = image.copy()
    for point, label in zip(points, labels):
        x, y = int(point[0]), int(point[1])
        color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Yeşil: pozitif, Kırmızı: negatif
        cv2.circle(result, (x, y), 5, color, -1)
        cv2.circle(result, (x, y), 7, (255, 255, 255), 2)
    return result


def visualize_box(image, box):
    """Kutuyu görüntü üzerine çizer."""
    result = image.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return result


def track_object_with_points(
    video_dir,
    frame_names,
    predictor,
    inference_state,
    ann_frame_idx=0,
    points=None,
    labels=None,
    obj_id=1
):
    """
    Nokta tıklamalarını kullanarak nesneyi takip eder.

    Args:
        video_dir: Frame'lerin bulunduğu dizin
        frame_names: Frame dosya isimleri listesi
        predictor: EdgeTAM video predictor
        inference_state: Inference state
        ann_frame_idx: Etkileşim yapılacak frame indexi
        points: Nokta koordinatları (N, 2) array
        labels: Nokta etiketleri (1: pozitif, 0: negatif)
        obj_id: Nesne ID'si

    Returns:
        video_segments: Her frame için segmentasyon sonuçları
    """
    # Varsayılan değerler
    if points is None:
        # Örnek: görüntünün merkezinde bir nokta
        first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
        h, w = first_frame.shape[:2]
        points = np.array([[w//2, h//2]], dtype=np.float32)

    if labels is None:
        labels = np.array([1], np.int32)  # Pozitif tıklama

    print(f"\nFrame {ann_frame_idx} üzerinde nesne seçiliyor...")
    print(f"Noktalar: {points}")
    print(f"Etiketler: {labels}")

    # İlk frame'de nesneyi seç
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    # İlk sonucu görselleştir
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[ann_frame_idx]))
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

    result_img = show_mask_on_image(first_frame_rgb, first_mask, obj_id)
    result_img = visualize_points(result_img, points, labels)

    print("İlk segmentasyon tamamlandı.")

    # Video boyunca yayılım (propagation)
    print("\nVideo boyunca nesne takip ediliyor...")
    video_segments = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print(f"Takip tamamlandı. Toplam {len(video_segments)} frame işlendi.")

    return video_segments


def track_object_with_box(
    video_dir,
    frame_names,
    predictor,
    inference_state,
    ann_frame_idx=0,
    box=None,
    obj_id=1
):
    """
    Kutu (bounding box) kullanarak nesneyi takip eder.

    Args:
        video_dir: Frame'lerin bulunduğu dizin
        frame_names: Frame dosya isimleri listesi
        predictor: EdgeTAM video predictor
        inference_state: Inference state
        ann_frame_idx: Etkileşim yapılacak frame indexi
        box: Kutu koordinatları [x_min, y_min, x_max, y_max]
        obj_id: Nesne ID'si

    Returns:
        video_segments: Her frame için segmentasyon sonuçları
    """
    # Varsayılan kutu
    if box is None:
        first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
        h, w = first_frame.shape[:2]
        # Görüntünün ortasında bir kutu
        box = np.array([w//4, h//4, 3*w//4, 3*h//4], dtype=np.float32)

    print(f"\nFrame {ann_frame_idx} üzerinde nesne seçiliyor...")
    print(f"Kutu: {box}")

    # İlk frame'de nesneyi seç
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=obj_id,
        box=box,
    )

    # İlk sonucu görselleştir
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[ann_frame_idx]))
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

    result_img = show_mask_on_image(first_frame_rgb, first_mask, obj_id)
    result_img = visualize_box(result_img, box)

    print("İlk segmentasyon tamamlandı.")

    # Video boyunca yayılım
    print("\nVideo boyunca nesne takip ediliyor...")
    video_segments = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print(f"Takip tamamlandı. Toplam {len(video_segments)} frame işlendi.")

    return video_segments


def save_tracking_results(video_dir, frame_names, video_segments, output_path):
    """Takip sonuçlarını video olarak kaydeder."""
    # İlk frame'i oku ve boyutları al
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    height, width = first_frame.shape[:2]

    # Video writer oluştur
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nSonuçlar kaydediliyor: {output_path}")

    for frame_idx, frame_name in enumerate(frame_names):
        # Frame'i oku
        frame = cv2.imread(os.path.join(video_dir, frame_name))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Maskeleri ekle
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                mask = mask.squeeze()
                frame_rgb = show_mask_on_image(frame_rgb, mask, obj_id)

        # BGR'ye çevir ve kaydet
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video kaydedildi: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='EdgeTAM Object Tracking Örneği')
    parser.add_argument('--video_path', type=str, default='examples/01_dog.mp4',
                        help='Takip edilecek video dosyası')
    parser.add_argument('--output_path', type=str, default='output_tracking.mp4',
                        help='Çıktı video dosyası')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/edgetam.pt',
                        help='Model checkpoint dosyası')
    parser.add_argument('--model_cfg', type=str, default='configs/edgetam.yaml',
                        help='Model konfigürasyon dosyası')
    parser.add_argument('--mode', type=str, default='point', choices=['point', 'box'],
                        help='Takip modu: point (nokta) veya box (kutu)')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Nesne seçimi yapılacak frame indexi')

    args = parser.parse_args()

    print("="*60)
    print("EdgeTAM Object Tracking Örneği")
    print("="*60)

    # Geçici dizin oluştur
    temp_dir = "temp_frames"

    # Cihaz seçimi
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nKullanılan cihaz: {device}")

    # Autocast ayarla
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # EdgeTAM modelini yükle
    print(f"\nEdgeTAM modeli yükleniyor...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.model_cfg}")

    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=device)
    print("Model yüklendi.")

    # Video'dan frame'leri çıkar
    if args.video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        frame_names = extract_frames_from_video(args.video_path, temp_dir)
        video_dir = temp_dir
    else:
        # Dizin olarak verilmişse
        video_dir = args.video_path
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Inference state'i başlat
    print(f"\nInference state başlatılıyor...")
    inference_state = predictor.init_state(video_path=video_dir)

    # Nesne takibi
    if args.mode == 'point':
        print("\n" + "="*60)
        print("MOD: Nokta Tıklama ile Takip")
        print("="*60)

        # Örnek: İlk frame'in merkezinde bir nokta
        first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
        h, w = first_frame.shape[:2]

        # Birden fazla nokta da eklenebilir
        points = np.array([
            [w//2, h//2],  # Merkez
        ], dtype=np.float32)

        labels = np.array([1], np.int32)  # Pozitif tıklama

        video_segments = track_object_with_points(
            video_dir=video_dir,
            frame_names=frame_names,
            predictor=predictor,
            inference_state=inference_state,
            ann_frame_idx=args.frame_idx,
            points=points,
            labels=labels,
            obj_id=1
        )

    else:  # box mode
        print("\n" + "="*60)
        print("MOD: Kutu (Bounding Box) ile Takip")
        print("="*60)

        # Örnek: Görüntünün ortasında bir kutu
        first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
        h, w = first_frame.shape[:2]

        box = np.array([
            w//4,      # x_min
            h//4,      # y_min
            3*w//4,    # x_max
            3*h//4     # y_max
        ], dtype=np.float32)

        video_segments = track_object_with_box(
            video_dir=video_dir,
            frame_names=frame_names,
            predictor=predictor,
            inference_state=inference_state,
            ann_frame_idx=args.frame_idx,
            box=box,
            obj_id=1
        )

    # Sonuçları kaydet
    save_tracking_results(video_dir, frame_names, video_segments, args.output_path)

    # Geçici dosyaları temizle
    if os.path.exists(temp_dir) and args.video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\nGeçici dosyalar temizlendi: {temp_dir}")

    print("\n" + "="*60)
    print("Takip tamamlandı!")
    print(f"Çıktı: {args.output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
