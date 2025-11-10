"""
EdgeTAM ONNX Doğrulama Scripti
===============================

Bu script ONNX'e dönüştürülmüş EdgeTAM modelinin doğruluğunu test eder.
PyTorch ve ONNX modellerinin çıktılarını karşılaştırır.

Kullanım:
    python examples/validate_onnx.py --onnx_dir onnx_models --checkpoint checkpoints/edgetam.pt

Parametreler:
    --onnx_dir: ONNX modellerinin bulunduğu dizin
    --checkpoint: PyTorch model checkpoint
    --config: Model config dosyası
    --image: Test görüntüsü (opsiyonel)
    --tolerance: Hata toleransı (varsayılan: 1e-3)
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2


def load_onnx_model(onnx_path):
    """ONNX modelini yükle"""
    try:
        import onnxruntime as ort

        # Session oluştur
        session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )

        return session

    except ImportError:
        print("❌ Hata: onnxruntime yüklü değil")
        print("  Yüklemek için: pip install onnxruntime")
        return None
    except Exception as e:
        print(f"❌ ONNX model yükleme hatası: {e}")
        return None


def compare_outputs(pytorch_output, onnx_output, name="Output", tolerance=1e-3):
    """PyTorch ve ONNX çıktılarını karşılaştır"""
    print(f"\n{'='*60}")
    print(f"Karşılaştırma: {name}")
    print(f"{'='*60}")

    # Numpy'a dönüştür
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_np = pytorch_output.detach().cpu().numpy()
    else:
        pytorch_np = np.array(pytorch_output)

    if isinstance(onnx_output, list):
        onnx_np = onnx_output[0]
    else:
        onnx_np = onnx_output

    # Shape kontrolü
    print(f"PyTorch shape: {pytorch_np.shape}")
    print(f"ONNX shape: {onnx_np.shape}")

    if pytorch_np.shape != onnx_np.shape:
        print(f"❌ Shape uyuşmuyor!")
        return False

    # İstatistikler
    diff = np.abs(pytorch_np - onnx_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    print(f"\nİstatistikler:")
    print(f"  Max fark: {max_diff:.6f}")
    print(f"  Mean fark: {mean_diff:.6f}")
    print(f"  Std fark: {std_diff:.6f}")
    print(f"  Tolerance: {tolerance}")

    # Değer aralıkları
    print(f"\nDeğer aralıkları:")
    print(f"  PyTorch: [{np.min(pytorch_np):.6f}, {np.max(pytorch_np):.6f}]")
    print(f"  ONNX: [{np.min(onnx_np):.6f}, {np.max(onnx_np):.6f}]")

    # Cosine similarity
    pytorch_flat = pytorch_np.flatten()
    onnx_flat = onnx_np.flatten()

    dot_product = np.dot(pytorch_flat, onnx_flat)
    norm_pytorch = np.linalg.norm(pytorch_flat)
    norm_onnx = np.linalg.norm(onnx_flat)

    cosine_sim = dot_product / (norm_pytorch * norm_onnx + 1e-8)
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    # Doğruluk kontrolü
    if max_diff < tolerance:
        print(f"\n✅ BAŞARILI: Max fark tolerance içinde")
        success = True
    else:
        print(f"\n❌ BAŞARISIZ: Max fark tolerance'ı aşıyor")
        success = False

    # Yakın değerlerin yüzdesi
    close_values = np.sum(diff < tolerance)
    total_values = diff.size
    close_percentage = (close_values / total_values) * 100
    print(f"  Tolerance içindeki değerler: {close_percentage:.2f}%")

    return success


def validate_image_encoder(pytorch_model, onnx_session, image_size=1024, tolerance=1e-3):
    """Image Encoder'ı doğrula"""
    print("\n" + "="*60)
    print("Image Encoder Doğrulaması")
    print("="*60)

    # Test input oluştur
    dummy_image = torch.randn(1, 3, image_size, image_size)

    print(f"\nTest input shape: {dummy_image.shape}")

    # PyTorch inference
    print("\n🔧 PyTorch inference...")
    with torch.no_grad():
        pytorch_model.image_encoder.eval()
        pytorch_output = pytorch_model.image_encoder(dummy_image)

    # ONNX inference
    print("🔧 ONNX inference...")
    onnx_input = {onnx_session.get_inputs()[0].name: dummy_image.numpy()}
    onnx_outputs = onnx_session.run(None, onnx_input)

    # Karşılaştır - ONNX 3 output döndürüyor: vision_features, feature_0, feature_1
    # PyTorch dict döndürüyor
    pytorch_vision_features = pytorch_output["vision_features"]

    # High-res features'ı ONNX wrapper'daki gibi işle
    if pytorch_model.use_high_res_features_in_sam:
        pytorch_feature_0 = pytorch_model.sam_mask_decoder.conv_s0(pytorch_output["backbone_fpn"][0])
        pytorch_feature_1 = pytorch_model.sam_mask_decoder.conv_s1(pytorch_output["backbone_fpn"][1])
    else:
        pytorch_feature_0 = pytorch_output["backbone_fpn"][0]
        pytorch_feature_1 = pytorch_output["backbone_fpn"][1]

    # Vision features karşılaştır
    success_vision = compare_outputs(
        pytorch_vision_features,
        onnx_outputs[0],
        name="Vision Features",
        tolerance=tolerance
    )

    # Feature 0 karşılaştır
    success_feat0 = compare_outputs(
        pytorch_feature_0,
        onnx_outputs[1],
        name="High-Res Feature 0",
        tolerance=tolerance
    )

    # Feature 1 karşılaştır
    success_feat1 = compare_outputs(
        pytorch_feature_1,
        onnx_outputs[2],
        name="High-Res Feature 1",
        tolerance=tolerance
    )

    return success_vision and success_feat0 and success_feat1


def validate_mask_decoder(pytorch_model, onnx_session, tolerance=1e-3):
    """Mask Decoder'ı doğrula"""
    print("\n" + "="*60)
    print("Mask Decoder Doğrulaması")
    print("="*60)

    # Test inputs oluştur
    embed_dim = pytorch_model.sam_mask_decoder.transformer_dim
    embed_size = pytorch_model.image_size // 16

    dummy_image_embeddings = torch.randn(1, embed_dim, embed_size, embed_size)
    dummy_sparse_embeddings = torch.randn(1, 2, embed_dim)
    dummy_dense_embeddings = torch.randn(1, embed_dim, embed_size, embed_size)

    # High-res features - bunlar image encoder'dan RAW olarak gelir
    # Wrapper içinde conv_s0 ve conv_s1 ile işlenir
    dummy_high_res_raw_0 = torch.randn(1, embed_dim, embed_size * 4, embed_size * 4)
    dummy_high_res_raw_1 = torch.randn(1, embed_dim, embed_size * 2, embed_size * 2)

    # Processed high-res features (ONNX input olarak kullanılacak)
    if pytorch_model.use_high_res_features_in_sam:
        dummy_high_res_feature_0 = pytorch_model.sam_mask_decoder.conv_s0(dummy_high_res_raw_0)
        dummy_high_res_feature_1 = pytorch_model.sam_mask_decoder.conv_s1(dummy_high_res_raw_1)
    else:
        dummy_high_res_feature_0 = dummy_high_res_raw_0
        dummy_high_res_feature_1 = dummy_high_res_raw_1

    print(f"\nTest input shapes:")
    print(f"  Image embeddings: {dummy_image_embeddings.shape}")
    print(f"  Sparse embeddings: {dummy_sparse_embeddings.shape}")
    print(f"  Dense embeddings: {dummy_dense_embeddings.shape}")
    print(f"  High-res feature 0 (processed): {dummy_high_res_feature_0.shape}")
    print(f"  High-res feature 1 (processed): {dummy_high_res_feature_1.shape}")

    # PyTorch inference
    print("\n🔧 PyTorch inference...")
    with torch.no_grad():
        pytorch_model.sam_mask_decoder.eval()

        # High-res features'ı hazırla - artık zaten processed
        if pytorch_model.use_high_res_features_in_sam:
            high_res_features = [dummy_high_res_feature_0, dummy_high_res_feature_1]
        else:
            high_res_features = None

        pytorch_masks, pytorch_iou, _, _ = pytorch_model.sam_mask_decoder(
            image_embeddings=dummy_image_embeddings,
            image_pe=pytorch_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=dummy_sparse_embeddings,
            dense_prompt_embeddings=dummy_dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

    # ONNX inference - processed features kullan
    print("🔧 ONNX inference...")
    onnx_inputs = {
        onnx_session.get_inputs()[0].name: dummy_image_embeddings.numpy(),
        onnx_session.get_inputs()[1].name: dummy_sparse_embeddings.numpy(),
        onnx_session.get_inputs()[2].name: dummy_dense_embeddings.numpy(),
        onnx_session.get_inputs()[3].name: dummy_high_res_feature_0.numpy(),
        onnx_session.get_inputs()[4].name: dummy_high_res_feature_1.numpy(),
    }
    onnx_outputs = onnx_session.run(None, onnx_inputs)

    # Masks karşılaştır
    masks_success = compare_outputs(
        pytorch_masks,
        onnx_outputs[0],
        name="Mask Predictions",
        tolerance=tolerance
    )

    # IoU karşılaştır
    iou_success = compare_outputs(
        pytorch_iou,
        onnx_outputs[1],
        name="IoU Predictions",
        tolerance=tolerance
    )

    return masks_success and iou_success


def visualize_comparison(pytorch_output, onnx_output, output_path="comparison.png"):
    """PyTorch ve ONNX çıktılarını görselleştir"""
    try:
        # Numpy'a dönüştür
        if isinstance(pytorch_output, torch.Tensor):
            pytorch_np = pytorch_output.detach().cpu().numpy()
        else:
            pytorch_np = np.array(pytorch_output)

        if isinstance(onnx_output, list):
            onnx_np = onnx_output[0]
        else:
            onnx_np = onnx_output

        # İlk sample'ı al
        if len(pytorch_np.shape) > 2:
            pytorch_vis = pytorch_np[0, 0] if pytorch_np.shape[0] > 0 else pytorch_np[0]
            onnx_vis = onnx_np[0, 0] if onnx_np.shape[0] > 0 else onnx_np[0]
        else:
            pytorch_vis = pytorch_np
            onnx_vis = onnx_np

        # Fark hesapla
        diff = np.abs(pytorch_vis - onnx_vis)

        # Görselleştir
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(pytorch_vis, cmap='viridis')
        axes[0].set_title('PyTorch Output')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(onnx_vis, cmap='viridis')
        axes[1].set_title('ONNX Output')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Karşılaştırma görseli kaydedildi: {output_path}")

    except Exception as e:
        print(f"⚠ Görselleştirme hatası: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='EdgeTAM ONNX modelini doğrula',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
  # Temel doğrulama
  python examples/validate_onnx.py --onnx_dir onnx_models --checkpoint checkpoints/edgetam.pt

  # Özel tolerance ile
  python examples/validate_onnx.py --onnx_dir onnx_models --checkpoint checkpoints/edgetam.pt --tolerance 1e-4

  # Görselleştirme ile
  python examples/validate_onnx.py --onnx_dir onnx_models --checkpoint checkpoints/edgetam.pt --visualize
        """
    )

    parser.add_argument('--onnx_dir', type=str, default='onnx_models',
                       help='ONNX modellerinin bulunduğu dizin')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/edgetam.pt',
                       help='PyTorch model checkpoint')
    parser.add_argument('--config', type=str, default='configs/edgetam.yaml',
                       help='Model config dosyası')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                       help='Hata toleransı (varsayılan: 1e-3)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='PyTorch inference için cihaz')
    parser.add_argument('--visualize', action='store_true',
                       help='Karşılaştırmayı görselleştir')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("EdgeTAM ONNX Doğrulama")
    print("="*60)

    # ONNX model yolları
    image_encoder_path = os.path.join(args.onnx_dir, "edgetam_image_encoder.onnx")
    mask_decoder_path = os.path.join(args.onnx_dir, "edgetam_mask_decoder.onnx")

    # ONNX modellerin varlığını kontrol et
    if not os.path.exists(image_encoder_path):
        print(f"❌ Hata: Image Encoder ONNX modeli bulunamadı: {image_encoder_path}")
        print(f"\nÖnce ONNX dönüşümü yapın:")
        print(f"  python examples/convert_to_onnx.py --checkpoint {args.checkpoint} --output_dir {args.onnx_dir}")
        return

    if not os.path.exists(mask_decoder_path):
        print(f"❌ Hata: Mask Decoder ONNX modeli bulunamadı: {mask_decoder_path}")
        return

    # PyTorch model yükle
    print(f"\n🔧 PyTorch modeli yükleniyor...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: {args.config}")

    device = torch.device(args.device)
    pytorch_model = build_sam2(args.config, args.checkpoint, device=args.device)
    pytorch_model.eval()

    print(f"✓ PyTorch model yüklendi")

    # ONNX modelleri yükle
    print(f"\n🔧 ONNX modelleri yükleniyor...")

    image_encoder_session = load_onnx_model(image_encoder_path)
    if image_encoder_session is None:
        return

    mask_decoder_session = load_onnx_model(mask_decoder_path)
    if mask_decoder_session is None:
        return

    print(f"✓ ONNX modelleri yüklendi")

    # Doğrulama sonuçları
    results = {}

    # Image Encoder doğrula
    try:
        results['image_encoder'] = validate_image_encoder(
            pytorch_model,
            image_encoder_session,
            image_size=pytorch_model.image_size,
            tolerance=args.tolerance
        )
    except Exception as e:
        print(f"\n❌ Image Encoder doğrulama hatası: {e}")
        import traceback
        traceback.print_exc()
        results['image_encoder'] = False

    # Mask Decoder doğrula
    try:
        results['mask_decoder'] = validate_mask_decoder(
            pytorch_model,
            mask_decoder_session,
            tolerance=args.tolerance
        )
    except Exception as e:
        print(f"\n❌ Mask Decoder doğrulama hatası: {e}")
        import traceback
        traceback.print_exc()
        results['mask_decoder'] = False

    # Özet
    print("\n" + "="*60)
    print("DOĞRULAMA SONUÇLARI")
    print("="*60)

    all_success = True
    for name, success in results.items():
        status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
        print(f"  {name}: {status}")
        if not success:
            all_success = False

    print("\n" + "="*60)
    if all_success:
        print("✅ TÜM TESTLER BAŞARILI!")
        print("="*60)
        print("\nONNX modelleri production için hazır.")
        print(f"\nModel dosyaları:")
        print(f"  - {image_encoder_path}")
        print(f"  - {mask_decoder_path}")
    else:
        print("❌ BAZI TESTLER BAŞARISIZ!")
        print("="*60)
        print(f"\nÖneriler:")
        print(f"  1. Tolerance değerini artırın (--tolerance)")
        print(f"  2. ONNX opset versiyonunu değiştirin")
        print(f"  3. Model export ayarlarını kontrol edin")

    print()


if __name__ == "__main__":
    main()
