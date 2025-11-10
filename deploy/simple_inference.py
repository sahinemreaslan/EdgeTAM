#!/usr/bin/env python3
"""
EdgeTAM Simple Inference Script

This script performs inference using EdgeTAM models in ONNX or TensorRT format.
Supports both point-based and box-based prompting.

Usage:
    # ONNX inference
    python deploy/simple_inference.py --onnx-dir onnx_models --image image.jpg --point 512,512

    # TensorRT inference
    python deploy/simple_inference.py --engine-dir tensorrt_models --image image.jpg --point 512,512

    # Box prompt
    python deploy/simple_inference.py --onnx-dir onnx_models --image image.jpg --box 100,100,500,500
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ONNXPredictor:
    """EdgeTAM predictor using ONNX runtime"""

    def __init__(self, model_dir, use_high_res=True):
        try:
            import onnxruntime as ort
        except ImportError:
            print("Error: onnxruntime not installed")
            print("Install with: pip install onnxruntime")
            sys.exit(1)

        self.model_dir = Path(model_dir)
        self.use_high_res = use_high_res

        # Load models
        print("Loading ONNX models...")
        encoder_path = self.model_dir / "edgetam_image_encoder.onnx"
        decoder_path = self.model_dir / "edgetam_mask_decoder.onnx"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found: {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder not found: {decoder_path}")

        # Create sessions
        self.encoder_session = ort.InferenceSession(
            str(encoder_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.decoder_session = ort.InferenceSession(
            str(decoder_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Get input/output names
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_names = [out.name for out in self.encoder_session.get_outputs()]

        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        self.decoder_output_names = [out.name for out in self.decoder_session.get_outputs()]

        print(f"✓ Models loaded successfully")
        print(f"  Encoder inputs: {self.encoder_input_name}")
        print(f"  Encoder outputs: {self.encoder_output_names}")
        print(f"  Decoder inputs: {self.decoder_input_names}")
        print(f"  Decoder outputs: {self.decoder_output_names}")

        # Check if high-res features are available
        self.has_high_res = len(self.encoder_output_names) > 1
        print(f"  High-res features: {self.has_high_res}")

    def preprocess_image(self, image):
        """
        Preprocess image for the model

        Args:
            image: PIL Image or numpy array

        Returns:
            preprocessed image tensor [1, 3, 1024, 1024]
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Resize to 1024x1024
        image = image.resize((1024, 1024), Image.BILINEAR)

        # Convert to numpy and normalize to [0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0

        # Convert to CHW format
        if len(image_np.shape) == 2:  # Grayscale
            image_np = np.stack([image_np] * 3, axis=0)
        else:  # RGB
            image_np = image_np.transpose(2, 0, 1)

        # Add batch dimension
        image_np = np.expand_dims(image_np, axis=0)

        return image_np

    def encode_image(self, image):
        """
        Encode image to embeddings

        Args:
            image: Preprocessed image tensor [1, 3, 1024, 1024]

        Returns:
            image_embeddings and optionally high_res features
        """
        outputs = self.encoder_session.run(
            self.encoder_output_names,
            {self.encoder_input_name: image}
        )

        if self.has_high_res and self.use_high_res:
            return outputs[0], outputs[1], outputs[2]  # embeddings, high_res_0, high_res_1
        else:
            return outputs[0], None, None

    def predict_mask(self, image_embeddings, points, labels, high_res_0=None, high_res_1=None):
        """
        Predict mask from embeddings and prompts

        Args:
            image_embeddings: [1, 256, 64, 64]
            points: [N, 2] array of point coordinates
            labels: [N] array of point labels (1=foreground, 0=background)
            high_res_0: [1, 32, 256, 256] (optional)
            high_res_1: [1, 64, 128, 128] (optional)

        Returns:
            masks: [1, 1, 1024, 1024]
            iou_predictions: [1, 1]
        """
        # Prepare inputs
        point_coords = np.array(points, dtype=np.float32).reshape(1, -1, 2)
        point_labels = np.array(labels, dtype=np.int32).reshape(1, -1)

        # Build decoder inputs
        if self.has_high_res and self.use_high_res and high_res_0 is not None:
            decoder_inputs = {
                'image_embeddings': image_embeddings,
                'point_coords': point_coords,
                'point_labels': point_labels,
                'high_res_feat_0': high_res_0,
                'high_res_feat_1': high_res_1,
            }
        else:
            decoder_inputs = {
                'image_embeddings': image_embeddings,
                'point_coords': point_coords,
                'point_labels': point_labels,
            }

        # Run decoder
        masks, iou_predictions = self.decoder_session.run(
            self.decoder_output_names,
            decoder_inputs
        )

        return masks, iou_predictions

    def predict(self, image, points, labels):
        """
        Full prediction pipeline

        Args:
            image: PIL Image or numpy array
            points: List of (x, y) tuples or array
            labels: List of labels or array (1=foreground, 0=background)

        Returns:
            mask: [1024, 1024] binary mask
            iou_score: IoU confidence score
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)

        # Encode
        image_embeddings, high_res_0, high_res_1 = self.encode_image(image_tensor)

        # Decode
        masks, iou_predictions = self.predict_mask(
            image_embeddings, points, labels, high_res_0, high_res_1
        )

        # Post-process
        mask = masks[0, 0]  # [1024, 1024]
        mask_binary = (mask > 0.5).astype(np.uint8)
        iou_score = iou_predictions[0, 0]

        return mask_binary, iou_score


class TensorRTPredictor:
    """EdgeTAM predictor using TensorRT"""

    def __init__(self, model_dir, use_high_res=True):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            print("Error: TensorRT and PyCUDA not installed")
            print("Install with: pip install tensorrt pycuda")
            sys.exit(1)

        self.model_dir = Path(model_dir)
        self.use_high_res = use_high_res

        # TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Find engine files (try fp16 first, then fp32)
        encoder_path = self.model_dir / "edgetam_image_encoder_fp16.trt"
        decoder_path = self.model_dir / "edgetam_mask_decoder_fp16.trt"

        if not encoder_path.exists():
            encoder_path = self.model_dir / "edgetam_image_encoder.trt"
            decoder_path = self.model_dir / "edgetam_mask_decoder.trt"

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder engine not found in {self.model_dir}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder engine not found in {self.model_dir}")

        print(f"Loading TensorRT engines...")
        print(f"  Encoder: {encoder_path}")
        print(f"  Decoder: {decoder_path}")

        # Load engines
        self.encoder_engine = self._load_engine(encoder_path)
        self.decoder_engine = self._load_engine(decoder_path)

        # Create execution contexts
        self.encoder_context = self.encoder_engine.create_execution_context()
        self.decoder_context = self.decoder_engine.create_execution_context()

        print("✓ TensorRT engines loaded successfully")

        # Check for high-res features
        self.has_high_res = self.encoder_engine.num_bindings > 2

    def _load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            return runtime.deserialize_cuda_engine(f.read())

    def preprocess_image(self, image):
        """Preprocess image (same as ONNX)"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = image.resize((1024, 1024), Image.BILINEAR)
        image_np = np.array(image).astype(np.float32) / 255.0

        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=0)
        else:
            image_np = image_np.transpose(2, 0, 1)

        image_np = np.expand_dims(image_np, axis=0)
        return np.ascontiguousarray(image_np)

    def predict(self, image, points, labels):
        """
        Full prediction pipeline using TensorRT

        Note: TensorRT inference requires more complex memory management.
        For production use, consider using a proper TensorRT inference wrapper.
        """
        print("⚠ TensorRT inference is partially implemented")
        print("  For production use, consider using ONNX Runtime with TensorRT execution provider")
        print("  or implement full TensorRT inference with proper memory management")

        # Fall back to showing the structure
        raise NotImplementedError("Full TensorRT inference with memory management not yet implemented")


def visualize_result(image, mask, output_path, points=None):
    """
    Visualize segmentation result

    Args:
        image: Original image (PIL or numpy)
        mask: Binary mask [H, W]
        output_path: Path to save visualization
        points: Optional list of (x, y) points to visualize
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Resize mask to match image size if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create colored mask overlay
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [0, 255, 0]  # Green

    # Blend with original image
    alpha = 0.5
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    # Draw points if provided
    if points is not None:
        for px, py in points:
            cv2.circle(blended, (int(px), int(py)), 5, (255, 0, 0), -1)
            cv2.circle(blended, (int(px), int(py)), 7, (255, 255, 255), 2)

    # Save result
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"✓ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EdgeTAM Simple Inference")
    parser.add_argument(
        "--onnx-dir",
        type=str,
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--engine-dir",
        type=str,
        help="Directory containing TensorRT engines",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "--point",
        type=str,
        help="Point prompt as 'x,y' (can specify multiple with --point x1,y1 --point x2,y2)",
        action="append",
    )
    parser.add_argument(
        "--box",
        type=str,
        help="Box prompt as 'x1,y1,x2,y2'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--no-high-res",
        action="store_true",
        help="Disable high-resolution features",
    )

    args = parser.parse_args()

    # Check backend
    if args.onnx_dir and args.engine_dir:
        print("Error: Specify either --onnx-dir or --engine-dir, not both")
        sys.exit(1)

    if not args.onnx_dir and not args.engine_dir:
        print("Error: Must specify either --onnx-dir or --engine-dir")
        sys.exit(1)

    # Check prompts
    if not args.point and not args.box:
        print("Error: Must specify either --point or --box prompt")
        sys.exit(1)

    # Load image
    print(f"\nLoading image: {args.image}")
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    orig_size = image.size
    print(f"✓ Image loaded: {orig_size[0]}x{orig_size[1]}")

    # Parse prompts
    points = []
    labels = []

    if args.point:
        for point_str in args.point:
            x, y = map(float, point_str.split(','))
            # Scale point to 1024x1024
            x_scaled = x * (1024 / orig_size[0])
            y_scaled = y * (1024 / orig_size[1])
            points.append([x_scaled, y_scaled])
            labels.append(1)  # Foreground
        print(f"✓ Point prompts: {len(points)} points")

    if args.box:
        x1, y1, x2, y2 = map(float, args.box.split(','))
        # Scale box to 1024x1024
        x1_scaled = x1 * (1024 / orig_size[0])
        y1_scaled = y1 * (1024 / orig_size[1])
        x2_scaled = x2 * (1024 / orig_size[0])
        y2_scaled = y2 * (1024 / orig_size[1])
        # Convert box to corner points
        points.extend([[x1_scaled, y1_scaled], [x2_scaled, y2_scaled]])
        labels.extend([2, 3])  # Box corner labels
        print(f"✓ Box prompt: ({x1}, {y1}) -> ({x2}, {y2})")

    # Create predictor
    print("\n" + "=" * 60)
    use_high_res = not args.no_high_res

    if args.onnx_dir:
        print("Using ONNX Runtime backend")
        predictor = ONNXPredictor(args.onnx_dir, use_high_res=use_high_res)
    else:
        print("Using TensorRT backend")
        predictor = TensorRTPredictor(args.engine_dir, use_high_res=use_high_res)

    print("=" * 60)

    # Run inference
    print("\nRunning inference...")
    start_time = time.time()

    try:
        mask, iou_score = predictor.predict(image, points, labels)
        inference_time = time.time() - start_time

        print(f"✓ Inference completed in {inference_time*1000:.2f}ms")
        print(f"  IoU confidence: {iou_score:.4f}")
        print(f"  Mask coverage: {(mask > 0).sum() / mask.size * 100:.2f}%")

        # Visualize
        visualize_result(
            image,
            mask,
            args.output,
            points=[[p[0] * orig_size[0] / 1024, p[1] * orig_size[1] / 1024] for p in points]
        )

        print("\n" + "=" * 60)
        print("✓ Inference completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
