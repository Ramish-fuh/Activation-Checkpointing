"""
Tensor Classifier: Classifies tensors as parameters, gradients, activations, etc.
"""

import torch
from typing import Tuple


class TensorClassifier:
    """
    Classifies tensors into categories: parameter, gradient, activation,
    optimizer_state, or other.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize classifier with model information.
        
        Args:
            model: PyTorch model
        """
        self.model = model
        self.parameter_shapes = set()
        self.parameter_ids = set()
        
        # Collect all parameter shapes and IDs
        for param in model.parameters():
            self.parameter_shapes.add(tuple(param.shape))
            self.parameter_ids.add(id(param))
    
    def classify_tensor(self, shape: Tuple[int, ...], 
                       requires_grad: bool, is_leaf: bool) -> str:
        """
        Classify a tensor based on its properties.
        
        Args:
            shape: Tensor shape
            requires_grad: Whether tensor requires gradients
            is_leaf: Whether tensor is a leaf in autograd graph
            
        Returns:
            Classification string: 'parameter', 'gradient', 'activation',
            'optimizer_state', or 'other'
        """
        # Parameters: leaf tensors with requires_grad that match model parameters
        if is_leaf and requires_grad and shape in self.parameter_shapes:
            return "parameter"
        
        # Gradients: typically have .grad attribute set
        # Note: We'll need to check this differently in actual implementation
        
        # Activations: intermediate tensors that require grad but aren't leaves
        if requires_grad and not is_leaf:
            return "activation"
        
        # Default
        return "other"
    
    def is_parameter(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is a model parameter."""
        return id(tensor) in self.parameter_ids
    
    def is_activation(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is an activation (intermediate computation)."""
        return tensor.requires_grad and not tensor.is_leaf
