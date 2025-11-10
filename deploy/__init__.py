"""
EdgeTAM Deployment Package

This package provides tools for deploying EdgeTAM models in production.
"""

__version__ = "1.0.0"

# Import main classes for easier access
try:
    from .simple_inference import ONNXPredictor, TensorRTPredictor
    __all__ = ['ONNXPredictor', 'TensorRTPredictor']
except ImportError:
    # Dependencies might not be installed
    __all__ = []
