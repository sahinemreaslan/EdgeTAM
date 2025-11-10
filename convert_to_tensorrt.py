#!/usr/bin/env python3
"""
EdgeTAM TensorRT Conversion Script

This script converts ONNX models to TensorRT engines for optimized inference.
Supports both FP32 and FP16 precision modes.

Usage:
    python convert_to_tensorrt.py --onnx-dir onnx_models --output-dir tensorrt_models
    python convert_to_tensorrt.py --onnx-dir onnx_models --fp16  # Enable FP16 mode
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"Error: {e}")
    print("\nTensorRT and PyCUDA are required for this script.")
    print("Please install them:")
    print("  pip install tensorrt pycuda")
    print("\nNote: TensorRT requires CUDA and cuDNN to be installed.")
    print("Visit: https://developer.nvidia.com/tensorrt")
    sys.exit(1)


# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(
    onnx_path,
    engine_path,
    fp16_mode=False,
    int8_mode=False,
    max_batch_size=1,
    workspace_size=4,
    verbose=False,
):
    """
    Build TensorRT engine from ONNX model

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        fp16_mode: Enable FP16 precision mode
        int8_mode: Enable INT8 precision mode
        max_batch_size: Maximum batch size
        workspace_size: Maximum workspace size in GB
        verbose: Enable verbose logging

    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        TRT_LOGGER.min_severity = trt.Logger.VERBOSE

    print(f"\n{'=' * 60}")
    print(f"Building TensorRT Engine")
    print(f"{'=' * 60}")
    print(f"ONNX Model: {onnx_path}")
    print(f"Output Engine: {engine_path}")
    print(f"FP16 Mode: {fp16_mode}")
    print(f"INT8 Mode: {int8_mode}")
    print(f"Max Batch Size: {max_batch_size}")
    print(f"Workspace Size: {workspace_size}GB")
    print(f"{'=' * 60}\n")

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    print("[1/3] Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    print("✓ ONNX model parsed successfully")

    # Create builder config
    config = builder.create_builder_config()

    # Set workspace size
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_size * (1 << 30)  # Convert GB to bytes
    )

    # Enable precision modes
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")
    elif fp16_mode:
        print("⚠ FP16 mode requested but not supported on this platform")

    if int8_mode and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("✓ INT8 mode enabled")
        print("⚠ INT8 calibration not implemented, using default calibration")
    elif int8_mode:
        print("⚠ INT8 mode requested but not supported on this platform")

    # Build engine
    print(f"\n[2/3] Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return False

    print("✓ Engine built successfully")

    # Save engine
    print(f"\n[3/3] Saving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    # Get file size in MB
    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✓ Engine saved successfully ({file_size_mb:.2f} MB)")

    return True


def verify_engine(engine_path):
    """
    Verify TensorRT engine

    Args:
        engine_path: Path to TensorRT engine

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            print(f"✗ Failed to load engine: {engine_path}")
            return False

        print(f"\n{'=' * 60}")
        print(f"Engine Verification: {engine_path}")
        print(f"{'=' * 60}")

        # Get engine info
        print(f"Number of bindings: {engine.num_bindings}")
        print(f"Number of optimization profiles: {engine.num_optimization_profiles}")

        # Print input/output information
        print(f"\nInputs:")
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                name = engine.get_binding_name(i)
                dtype = engine.get_binding_dtype(i)
                shape = engine.get_binding_shape(i)
                print(f"  {name}: {dtype} {shape}")

        print(f"\nOutputs:")
        for i in range(engine.num_bindings):
            if not engine.binding_is_input(i):
                name = engine.get_binding_name(i)
                dtype = engine.get_binding_dtype(i)
                shape = engine.get_binding_shape(i)
                print(f"  {name}: {dtype} {shape}")

        print(f"{'=' * 60}")
        print("✓ Engine verification passed")

        return True

    except Exception as e:
        print(f"✗ Engine verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert EdgeTAM ONNX models to TensorRT engines"
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="onnx_models",
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tensorrt_models",
        help="Output directory for TensorRT engines",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision mode (faster, slightly less accurate)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 precision mode (fastest, requires calibration)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="Maximum workspace size in GB (default: 4)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size (default: 1)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify TensorRT engines after conversion",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Check if ONNX directory exists
    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.exists():
        print(f"Error: ONNX directory not found: {onnx_dir}")
        print("Please run export_to_onnx.py first to generate ONNX models")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ONNX models
    encoder_onnx = onnx_dir / "edgetam_image_encoder.onnx"
    decoder_onnx = onnx_dir / "edgetam_mask_decoder.onnx"

    if not encoder_onnx.exists():
        print(f"Error: Image encoder ONNX not found: {encoder_onnx}")
        sys.exit(1)

    if not decoder_onnx.exists():
        print(f"Error: Mask decoder ONNX not found: {decoder_onnx}")
        sys.exit(1)

    # Determine engine suffix
    suffix = ""
    if args.fp16:
        suffix = "_fp16"
    elif args.int8:
        suffix = "_int8"

    # Convert image encoder
    print("\n" + "=" * 60)
    print("Converting Image Encoder")
    print("=" * 60)

    encoder_engine = output_dir / f"edgetam_image_encoder{suffix}.trt"
    encoder_success = build_engine(
        str(encoder_onnx),
        str(encoder_engine),
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        max_batch_size=args.max_batch_size,
        workspace_size=args.workspace,
        verbose=args.verbose,
    )

    if not encoder_success:
        print("✗ Failed to convert image encoder")
        sys.exit(1)

    # Convert mask decoder
    print("\n" + "=" * 60)
    print("Converting Mask Decoder")
    print("=" * 60)

    decoder_engine = output_dir / f"edgetam_mask_decoder{suffix}.trt"
    decoder_success = build_engine(
        str(decoder_onnx),
        str(decoder_engine),
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        max_batch_size=args.max_batch_size,
        workspace_size=args.workspace,
        verbose=args.verbose,
    )

    if not decoder_success:
        print("✗ Failed to convert mask decoder")
        sys.exit(1)

    # Verify engines
    if args.verify:
        print("\n" + "=" * 60)
        print("Verifying TensorRT Engines")
        print("=" * 60)

        verify_engine(str(encoder_engine))
        verify_engine(str(decoder_engine))

    # Summary
    print("\n" + "=" * 60)
    print("✓ TensorRT Conversion Complete!")
    print("=" * 60)
    print(f"Image Encoder: {encoder_engine}")
    print(f"Mask Decoder: {decoder_engine}")
    print("=" * 60)

    # Print next steps
    print("\nNext steps:")
    print("1. Verify engines:")
    print(f"   python convert_to_tensorrt.py --onnx-dir {args.onnx_dir} --verify")
    print("\n2. Run inference:")
    print(f"   python deploy/simple_inference.py --engine-dir {args.output_dir}")
    print("\n3. Benchmark performance:")
    print(f"   python deploy/benchmark.py --engine-dir {args.output_dir}")


if __name__ == "__main__":
    main()
