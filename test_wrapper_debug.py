"""
ImageEncoderWrapper debug script
Test whether the wrapper properly wraps image_encoder
"""
import torch
import sys
sys.path.insert(0, 'examples')

from sam2.build_sam import build_sam2
from convert_to_onnx import ImageEncoderWrapper

# Load model
model = build_sam2('configs/edgetam.yaml', 'checkpoints/edgetam.pt', device='cpu')
model.eval()

print("="*60)
print("Testing ImageEncoderWrapper")
print("="*60)

# Create wrapper
wrapper = ImageEncoderWrapper(model)
wrapper.eval()

print(f"\nWrapper modules:")
for name, module in wrapper.named_modules():
    if name:  # Skip root
        print(f"  {name}: {type(module).__name__}")

# Test forward
dummy_input = torch.randn(1, 3, 1024, 1024)

print(f"\nInput shape: {dummy_input.shape}")
print(f"Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")

with torch.no_grad():
    # Direct image_encoder call
    print("\n" + "="*60)
    print("Direct image_encoder.forward()")
    print("="*60)
    backbone_out = model.image_encoder(dummy_input)
    print(f"vision_features shape: {backbone_out['vision_features'].shape}")
    print(f"vision_features range: [{backbone_out['vision_features'].min():.6f}, {backbone_out['vision_features'].max():.6f}]")
    print(f"backbone_fpn[0] shape: {backbone_out['backbone_fpn'][0].shape}")
    print(f"backbone_fpn[0] range: [{backbone_out['backbone_fpn'][0].min():.6f}, {backbone_out['backbone_fpn'][0].max():.6f}]")
    print(f"backbone_fpn[1] shape: {backbone_out['backbone_fpn'][1].shape}")
    print(f"backbone_fpn[1] range: [{backbone_out['backbone_fpn'][1].min():.6f}, {backbone_out['backbone_fpn'][1].max():.6f}]")

    # Apply conv layers manually
    if model.use_high_res_features_in_sam:
        feat0 = model.sam_mask_decoder.conv_s0(backbone_out['backbone_fpn'][0])
        feat1 = model.sam_mask_decoder.conv_s1(backbone_out['backbone_fpn'][1])
        print(f"\nAfter conv_s0: shape={feat0.shape}, range=[{feat0.min():.6f}, {feat0.max():.6f}]")
        print(f"After conv_s1: shape={feat1.shape}, range=[{feat1.min():.6f}, {feat1.max():.6f}]")

    # Wrapper call
    print("\n" + "="*60)
    print("ImageEncoderWrapper.forward()")
    print("="*60)
    vision_features, feature_0, feature_1 = wrapper(dummy_input)
    print(f"vision_features shape: {vision_features.shape}")
    print(f"vision_features range: [{vision_features.min():.6f}, {vision_features.max():.6f}]")
    print(f"feature_0 shape: {feature_0.shape}")
    print(f"feature_0 range: [{feature_0.min():.6f}, {feature_0.max():.6f}]")
    print(f"feature_1 shape: {feature_1.shape}")
    print(f"feature_1 range: [{feature_1.min():.6f}, {feature_1.max():.6f}]")

print("\n" + "="*60)
print("Check if outputs match")
print("="*60)
vision_match = torch.allclose(backbone_out['vision_features'], vision_features)
print(f"Vision features match: {vision_match}")
if model.use_high_res_features_in_sam:
    feat0_match = torch.allclose(feat0, feature_0)
    feat1_match = torch.allclose(feat1, feature_1)
    print(f"Feature 0 match: {feat0_match}")
    print(f"Feature 1 match: {feat1_match}")

print("\nTest complete!")
