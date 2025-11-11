"""
EdgeTAM ONNX Dönüştürme Scripti
================================

Bu script EdgeTAM modelini ONNX formatına dönüştürür.
Model component'ler halinde export edilir:
1. Image Encoder
2. Prompt Encoder
3. Mask Decoder

Kullanım:
    python examples/convert_to_onnx.py --checkpoint checkpoints/edgetam.pt --output_dir onnx_models

Parametreler:
    --checkpoint: PyTorch model checkpoint
    --config: Model config dosyası
    --output_dir: ONNX modellerinin kaydedileceği dizin
    --opset_version: ONNX opset versiyonu (varsayılan: 17)
    --simplify: ONNX modelini simplify et (varsayılan: False)
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ImageEncoderWrapper(torch.nn.Module):
    """Image Encoder için ONNX export wrapper"""
    def __init__(self, model):
        super().__init__()
        self.image_encoder = model.image_encoder
        self.use_high_res_features = model.use_high_res_features_in_sam

        # EdgeTAM'de high-res features için conv layers var
        # SAM2Base'deki forward_image metodunda bunlar pre-compute ediliyor
        # IMPORTANT: add_module kullanarak ONNX exporter'ın görmesini sağla
        if self.use_high_res_features:
            # Deep copy yerine direkt module olarak ekle - ONNX trace için gerekli
            self.add_module('conv_s0', model.sam_mask_decoder.conv_s0)
            self.add_module('conv_s1', model.sam_mask_decoder.conv_s1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - Normalize edilmiş görüntü
        Returns:
            vision_features: Main feature map (B, C, H/16, W/16)
            feature_0: High-res feature level 0 processed (B, C/8, H/4, W/4)
            feature_1: High-res feature level 1 processed (B, C/4, H/8, W/8)
        """
        backbone_out = self.image_encoder(x)

        # Main vision features (en düşük resolution)
        vision_features = backbone_out["vision_features"]

        # High-resolution features for mask decoder
        backbone_fpn = backbone_out["backbone_fpn"]

        if self.use_high_res_features:
            # SAM2Base.forward_image metodundaki gibi pre-process et
            # Bu conv'lar mask decoder'da tekrar çağrılmasın diye burada yapılıyor
            feature_0 = self.conv_s0(backbone_fpn[0])  # Highest resolution
            feature_1 = self.conv_s1(backbone_fpn[1])  # Mid resolution
        else:
            feature_0 = backbone_fpn[0]
            feature_1 = backbone_fpn[1]

        return vision_features, feature_0, feature_1


class PromptEncoderWrapper(torch.nn.Module):
    """Prompt Encoder için ONNX export wrapper"""
    def __init__(self, model):
        super().__init__()
        self.sam_prompt_encoder = model.sam_prompt_encoder

    def forward(self, points, labels, boxes, masks):
        """
        Args:
            points: (B, N, 2) - Nokta koordinatları
            labels: (B, N) - Nokta etiketleri (1: pozitif, 0: negatif)
            boxes: (B, 4) - Bounding box [x1, y1, x2, y2]
            masks: (B, 1, H, W) - Mask input
        Returns:
            sparse_embeddings: Sparse prompt embeddings
            dense_embeddings: Dense prompt embeddings
        """
        # Points
        if points is not None and points.numel() > 0:
            point_coords = points
            point_labels = labels
        else:
            point_coords = None
            point_labels = None

        # Box
        if boxes is not None and boxes.numel() > 0:
            box_coords = boxes
        else:
            box_coords = None

        # Mask
        if masks is not None and masks.numel() > 0:
            mask_input = masks
        else:
            mask_input = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(point_coords, point_labels) if point_coords is not None else None,
            boxes=box_coords,
            masks=mask_input,
        )

        return sparse_embeddings, dense_embeddings


class MaskDecoderWrapper(torch.nn.Module):
    """Mask Decoder için ONNX export wrapper"""
    def __init__(self, model):
        super().__init__()
        self.sam_mask_decoder = model.sam_mask_decoder
        self.sam_prompt_encoder = model.sam_prompt_encoder
        self.image_size = model.image_size
        self.use_high_res_features = model.use_high_res_features_in_sam

        # Positional encoding'i önceden hesapla ve buffer olarak kaydet
        with torch.no_grad():
            self.register_buffer('image_pe', self.sam_prompt_encoder.get_dense_pe())

    def forward(self, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings,
                high_res_feature_0, high_res_feature_1):
        """
        Args:
            image_embeddings: Image encoder features (B, C, H, W)
            sparse_prompt_embeddings: Sparse prompt embeddings
            dense_prompt_embeddings: Dense prompt embeddings
            high_res_feature_0: Already processed high-res feature level 0 (B, C/8, 4*H, 4*W)
            high_res_feature_1: Already processed high-res feature level 1 (B, C/4, 2*H, 2*W)
        Returns:
            masks: (B, N, H, W) - Predicted masks
            iou_predictions: (B, N) - IoU predictions
        """
        # High-res features'ı hazırla
        # NOT: Features artık image encoder'dan pre-processed olarak geliyor
        if self.use_high_res_features:
            high_res_features = [high_res_feature_0, high_res_feature_1]
        else:
            high_res_features = None

        low_res_masks, iou_predictions, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks, iou_predictions


def export_image_encoder(model, output_path, opset_version=17):
    """Image Encoder'ı ONNX'e export et"""
    print("\n" + "="*60)
    print("Image Encoder ONNX'e dönüştürülüyor...")
    print("="*60)

    # Wrapper oluştur
    encoder = ImageEncoderWrapper(model)
    encoder.eval()

    # Örnek input
    image_size = model.image_size
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Input ve output isimleri
    input_names = ["image"]
    output_names = ["vision_features", "feature_0", "feature_1"]

    # Dynamic axes
    dynamic_axes = {
        "image": {0: "batch"},
        "vision_features": {0: "batch"},
        "feature_0": {0: "batch"},
        "feature_1": {0: "batch"},
    }

    print(f"Input shape: {dummy_input.shape}")
    print(f"Image size: {image_size}")

    # Export
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )

    print(f"✓ Image Encoder kaydedildi: {output_path}")
    print(f"  Opset version: {opset_version}")


def export_mask_decoder(model, output_path, opset_version=17):
    """Mask Decoder'ı ONNX'e export et"""
    print("\n" + "="*60)
    print("Mask Decoder ONNX'e dönüştürülüyor...")
    print("="*60)

    # Wrapper oluştur
    decoder = MaskDecoderWrapper(model)
    decoder.eval()

    # Örnek inputs
    # Image embeddings (encoder output'una benzer)
    embed_dim = model.sam_mask_decoder.transformer_dim
    embed_size = model.image_size // 16  # Typical downsampling

    dummy_image_embeddings = torch.randn(1, embed_dim, embed_size, embed_size)
    dummy_sparse_embeddings = torch.randn(1, 2, embed_dim)  # 2 prompts
    dummy_dense_embeddings = torch.randn(1, embed_dim, embed_size, embed_size)

    # High-resolution features - bunlar image encoder'dan processed olarak gelir
    # conv_s0 ve conv_s1 zaten image encoder wrapper'da uygulanmış olacak
    # feature_0: 4x resolution (256x256 for 1024x1024 input), processed to C/8 channels
    # feature_1: 2x resolution (128x128 for 1024x1024 input), processed to C/4 channels
    # SAM decoder conv_s0/conv_s1 output dimensions: transformer_dim // 8 and transformer_dim // 4
    if model.use_high_res_features_in_sam:
        # Processed dimensions
        dummy_high_res_feature_0 = torch.randn(1, embed_dim // 8, embed_size * 4, embed_size * 4)
        dummy_high_res_feature_1 = torch.randn(1, embed_dim // 4, embed_size * 2, embed_size * 2)
    else:
        # Raw dimensions (not used if use_high_res_features is False)
        dummy_high_res_feature_0 = torch.randn(1, embed_dim, embed_size * 4, embed_size * 4)
        dummy_high_res_feature_1 = torch.randn(1, embed_dim, embed_size * 2, embed_size * 2)

    # Input ve output isimleri
    input_names = [
        "image_embeddings",
        "sparse_prompt_embeddings",
        "dense_prompt_embeddings",
        "high_res_feature_0",
        "high_res_feature_1"
    ]
    output_names = ["masks", "iou_predictions"]

    # Dynamic axes
    dynamic_axes = {
        "image_embeddings": {0: "batch"},
        "sparse_prompt_embeddings": {0: "batch", 1: "num_prompts"},
        "dense_prompt_embeddings": {0: "batch"},
        "high_res_feature_0": {0: "batch"},
        "high_res_feature_1": {0: "batch"},
        "masks": {0: "batch"},
        "iou_predictions": {0: "batch"},
    }

    print(f"Image embeddings shape: {dummy_image_embeddings.shape}")
    print(f"Sparse embeddings shape: {dummy_sparse_embeddings.shape}")
    print(f"Dense embeddings shape: {dummy_dense_embeddings.shape}")
    print(f"High-res feature 0 shape: {dummy_high_res_feature_0.shape}")
    print(f"High-res feature 1 shape: {dummy_high_res_feature_1.shape}")

    # Export
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (dummy_image_embeddings, dummy_sparse_embeddings, dummy_dense_embeddings,
             dummy_high_res_feature_0, dummy_high_res_feature_1),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )

    print(f"✓ Mask Decoder kaydedildi: {output_path}")
    print(f"  Opset version: {opset_version}")


def simplify_onnx_model(onnx_path):
    """ONNX modelini simplify et (opsiyonel)"""
    try:
        import onnx
        from onnxsim import simplify

        print(f"\n📦 ONNX model simplify ediliyor: {onnx_path}")

        # Model yükle
        model = onnx.load(onnx_path)

        # Simplify
        model_simplified, check = simplify(model)

        if check:
            # Simplified model kaydet
            onnx.save(model_simplified, onnx_path)
            print(f"✓ Model simplified: {onnx_path}")
        else:
            print(f"⚠ Simplification başarısız oldu")

    except ImportError:
        print("⚠ onnx-simplifier yüklü değil. Simplify atlanıyor.")
        print("  Yüklemek için: pip install onnx-simplifier")
    except Exception as e:
        print(f"⚠ Simplify sırasında hata: {e}")


def get_model_info(onnx_path):
    """ONNX model hakkında bilgi göster"""
    try:
        import onnx

        model = onnx.load(onnx_path)

        # Model boyutu
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

        # Input/Output bilgileri
        print(f"\n📊 Model Bilgileri: {os.path.basename(onnx_path)}")
        print(f"  Dosya boyutu: {size_mb:.2f} MB")

        print(f"\n  Inputs:")
        for inp in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else "dynamic"
                    for dim in inp.type.tensor_type.shape.dim]
            print(f"    - {inp.name}: {shape}")

        print(f"\n  Outputs:")
        for out in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else "dynamic"
                    for dim in out.type.tensor_type.shape.dim]
            print(f"    - {out.name}: {shape}")

    except ImportError:
        print("⚠ onnx paketi yüklü değil. Model bilgisi gösterilemiyor.")
    except Exception as e:
        print(f"⚠ Model bilgisi alınırken hata: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='EdgeTAM modelini ONNX formatına dönüştür',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek kullanım:
  # Temel kullanım
  python examples/convert_to_onnx.py --checkpoint checkpoints/edgetam.pt --output_dir onnx_models

  # Simplify ile
  python examples/convert_to_onnx.py --checkpoint checkpoints/edgetam.pt --output_dir onnx_models --simplify

  # Özel opset version
  python examples/convert_to_onnx.py --checkpoint checkpoints/edgetam.pt --output_dir onnx_models --opset_version 18
        """
    )

    parser.add_argument('--checkpoint', type=str, default='checkpoints/edgetam.pt',
                       help='PyTorch model checkpoint')
    parser.add_argument('--config', type=str, default='configs/edgetam.yaml',
                       help='Model config dosyası')
    parser.add_argument('--output_dir', type=str, default='onnx_models',
                       help='ONNX modellerinin kaydedileceği dizin')
    parser.add_argument('--opset_version', type=int, default=17,
                       help='ONNX opset versiyonu (varsayılan: 17)')
    parser.add_argument('--simplify', action='store_true',
                       help='ONNX modelini simplify et (onnx-simplifier gerektirir)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Dönüştürme için kullanılacak cihaz')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("EdgeTAM ONNX Dönüştürme")
    print("="*60)

    # Checkpoint kontrolü
    if not os.path.exists(args.checkpoint):
        print(f"❌ Hata: Checkpoint bulunamadı: {args.checkpoint}")
        return

    # Output dizini oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 Output dizini: {args.output_dir}")

    # Model yükle
    print(f"\n🔧 EdgeTAM modeli yükleniyor...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: {args.config}")

    device = torch.device(args.device)
    model = build_sam2(args.config, args.checkpoint, device=args.device)
    model.eval()

    print(f"✓ Model yüklendi")
    print(f"  Image size: {model.image_size}")
    print(f"  Device: {device}")

    # Export paths
    image_encoder_path = os.path.join(args.output_dir, "edgetam_image_encoder.onnx")
    mask_decoder_path = os.path.join(args.output_dir, "edgetam_mask_decoder.onnx")

    # Image Encoder export
    try:
        export_image_encoder(
            model,
            image_encoder_path,
            opset_version=args.opset_version
        )

        # Model bilgilerini göster
        get_model_info(image_encoder_path)

        # Simplify (opsiyonel)
        if args.simplify:
            simplify_onnx_model(image_encoder_path)

    except Exception as e:
        print(f"❌ Image Encoder export hatası: {e}")
        import traceback
        traceback.print_exc()

    # Mask Decoder export
    try:
        export_mask_decoder(
            model,
            mask_decoder_path,
            opset_version=args.opset_version
        )

        # Model bilgilerini göster
        get_model_info(mask_decoder_path)

        # Simplify (opsiyonel)
        if args.simplify:
            simplify_onnx_model(mask_decoder_path)

    except Exception as e:
        print(f"❌ Mask Decoder export hatası: {e}")
        import traceback
        traceback.print_exc()

    # Özet
    print("\n" + "="*60)
    print("✅ ONNX Dönüştürme Tamamlandı!")
    print("="*60)
    print(f"\nDönüştürülen modeller:")
    if os.path.exists(image_encoder_path):
        size = os.path.getsize(image_encoder_path) / (1024 * 1024)
        print(f"  ✓ Image Encoder: {image_encoder_path} ({size:.2f} MB)")
    if os.path.exists(mask_decoder_path):
        size = os.path.getsize(mask_decoder_path) / (1024 * 1024)
        print(f"  ✓ Mask Decoder: {mask_decoder_path} ({size:.2f} MB)")

    print(f"\nSONRAKİ ADIM:")
    print(f"  Dönüşümü doğrulamak için:")
    print(f"  python examples/validate_onnx.py --onnx_dir {args.output_dir} --checkpoint {args.checkpoint}")
    print()


if __name__ == "__main__":
    main()
