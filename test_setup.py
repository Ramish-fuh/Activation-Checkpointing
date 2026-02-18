"""
Quick test script to verify the installation and basic functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
    except ImportError:
        print("[FAIL] PyTorch not installed")
        return False
    
    try:
        import torchvision
        print(f"[OK] Torchvision {torchvision.__version__}")
    except ImportError:
        print("[FAIL] Torchvision not installed")
        return False
    
    try:
        import transformers
        print(f"[OK] Transformers {transformers.__version__}")
    except ImportError:
        print("[FAIL] Transformers not installed")
        return False
    
    try:
        import matplotlib
        print(f"[OK] Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("[FAIL] Matplotlib not installed")
        return False
    
    try:
        import numpy
        print(f"[OK] NumPy {numpy.__version__}")
    except ImportError:
        print("[FAIL] NumPy not installed")
        return False
    
    return True


def test_models():
    """Test that models can be loaded."""
    print("\nTesting models...")
    
    try:
        from src.models.resnet152 import create_resnet152
        model = create_resnet152(num_classes=10, pretrained=False, device='cpu')
        print(f"[OK] ResNet-152 loaded successfully")
    except Exception as e:
        print(f"[FAIL] ResNet-152 failed: {e}")
        return False
    
    try:
        from src.models.bert import create_bert
        model = create_bert(model_name='bert-base-uncased', num_labels=2, 
                           pretrained=False, device='cpu')
        print(f"[OK] BERT loaded successfully")
    except Exception as e:
        print(f"[FAIL] BERT failed: {e}")
        return False
    
    return True


def test_profiler():
    """Test that profiler components can be imported."""
    print("\nTesting profiler components...")
    
    try:
        from src.phase1_profiler import (
            GraphProfiler, ComputationGraph, GraphNode,
            TensorClassifier, ActivationAnalyzer, MemoryVisualizer
        )
        print("[OK] All profiler components imported successfully")
    except Exception as e:
        print(f"[FAIL] Profiler import failed: {e}")
        return False
    
    return True


def test_simple_forward():
    """Test a simple forward pass."""
    print("\nTesting simple forward pass...")
    
    try:
        import torch
        from src.models.resnet152 import create_resnet152
        
        model = create_resnet152(num_classes=10, pretrained=False, device='cpu')
        dummy_input = torch.randn(2, 3, 224, 224)
        
        output = model(dummy_input)
        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
        
        print(f"[OK] Forward pass successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return False


def check_device_availability():
    """Check available devices."""
    print("\nChecking device availability...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[INFO] CUDA not available")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("[OK] MPS (Apple Silicon) available")
    else:
        print("[INFO] MPS not available")
    
    print("[OK] CPU available")


def main():
    """Run all tests."""
    print("="*70)
    print("CS265 Systems Project - Environment Check")
    print("="*70)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\nWARNING: Some packages are missing. Run: pip install -r requirements.txt")
        return
    
    # Check devices
    check_device_availability()
    
    # Test models
    if not test_models():
        all_passed = False
    
    # Test profiler
    if not test_profiler():
        all_passed = False
    
    # Test forward pass
    if not test_simple_forward():
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("[SUCCESS] All tests passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Run: python experiments/train_resnet.py")
        print("2. Run: python experiments/train_resnet.py --profile")
    else:
        print("[FAIL] Some tests failed. Please fix the issues above.")
    print("="*70)


if __name__ == '__main__':
    main()
