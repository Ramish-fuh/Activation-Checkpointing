"""
ResNet-152 Model Wrapper
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNet152Wrapper(nn.Module):
    """
    Wrapper for ResNet-152 model with configurable settings for profiling.
    """
    
    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        """
        Initialize ResNet-152 model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(ResNet152Wrapper, self).__init__()
        
        # Load ResNet-152
        if pretrained:
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            self.model = models.resnet152(weights=None)
        
        # Modify final layer if needed
        if num_classes != 1000:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-152."""
        return self.model(x)
    
    def get_model_info(self):
        """Return information about the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        
        return {
            'name': 'ResNet-152',
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_memory_mb': param_memory / (1024 ** 2),
            'num_classes': self.num_classes
        }
    
    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        print(f"\n{'='*60}")
        print(f"Model: {info['name']}")
        print(f"{'='*60}")
        print(f"Total Parameters: {info['total_params']:,}")
        print(f"Trainable Parameters: {info['trainable_params']:,}")
        print(f"Parameter Memory: {info['param_memory_mb']:.2f} MB")
        print(f"Number of Classes: {info['num_classes']}")
        print(f"{'='*60}\n")


def create_resnet152(num_classes: int = 1000, pretrained: bool = False, 
                     device: str = 'cpu') -> ResNet152Wrapper:
    """
    Factory function to create ResNet-152 model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to place model on ('cpu', 'cuda', 'mps')
        
    Returns:
        ResNet152Wrapper instance
    """
    model = ResNet152Wrapper(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model
