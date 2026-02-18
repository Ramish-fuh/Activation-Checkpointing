"""
Main Graph Profiler: Orchestrates graph capture and profiling.
"""

import torch
import time
from typing import Dict, List, Optional, Callable
from collections import defaultdict

from .graph_builder import ComputationGraph, GraphNode
from .tensor_classifier import TensorClassifier


class GraphProfiler:
    """
    Main profiler that captures the computation graph and profiles
    memory and compute statistics during a training iteration.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize the profiler.
        
        Args:
            model: PyTorch model to profile
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self.model = model
        self.device = device
        self.graph = ComputationGraph()
        self.classifier = TensorClassifier(model)
        
        # Hooks storage
        self.hooks = []
        
        # Track current operation
        self.current_node: Optional[GraphNode] = None
        self.operation_counter = 0
        
        # Memory tracking
        self.memory_snapshots = []
        
    def _register_hooks(self):
        """Register forward and backward hooks on all modules."""
        
        def forward_hook(module, inputs, outputs):
            """Hook called after forward pass of a module."""
            module_name = self._get_module_name(module)
            operation_type = module.__class__.__name__
            
            # Create node for this operation
            node = self.graph.add_node(
                operation_name=f"forward_{module_name}",
                operation_type=operation_type,
                is_forward=True,
                is_backward=False
            )
            
            # Record inputs
            if isinstance(inputs, tuple):
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        tensor_id = self.graph.add_tensor(inp)
                        self.graph.add_edge(None, node.node_id, tensor_id, is_input=True)
            
            # Record outputs
            if isinstance(outputs, torch.Tensor):
                tensor_id = self.graph.add_tensor(outputs)
                self.graph.add_edge(node.node_id, None, tensor_id, is_input=False)
            elif isinstance(outputs, tuple):
                for out in outputs:
                    if isinstance(out, torch.Tensor):
                        tensor_id = self.graph.add_tensor(out)
                        self.graph.add_edge(node.node_id, None, tensor_id, is_input=False)
            
            # Memory snapshot
            if self.device == 'cuda' and torch.cuda.is_available():
                node.memory_allocated = torch.cuda.memory_allocated()
            
            return outputs
        
        def backward_hook(module, grad_input, grad_output):
            """Hook called after backward pass of a module."""
            module_name = self._get_module_name(module)
            operation_type = module.__class__.__name__
            
            # Create node for backward operation
            node = self.graph.add_node(
                operation_name=f"backward_{module_name}",
                operation_type=operation_type,
                is_forward=False,
                is_backward=True
            )
            
            return grad_input
        
        # Register hooks on all modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                h1 = module.register_forward_hook(forward_hook)
                h2 = module.register_full_backward_hook(backward_hook)
                self.hooks.append(h1)
                self.hooks.append(h2)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _get_module_name(self, module: torch.nn.Module) -> str:
        """Get the name of a module."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name if name else "model"
        return "unknown"
    
    def profile_iteration(self, input_data: torch.Tensor, target: torch.Tensor, 
                         loss_fn: Callable, optimizer: torch.optim.Optimizer) -> ComputationGraph:
        """
        Profile a single training iteration.
        
        Args:
            input_data: Input tensor
            target: Target tensor  
            loss_fn: Loss function
            optimizer: Optimizer
            
        Returns:
            ComputationGraph with profiling information
        """
        # Reset graph
        self.graph = ComputationGraph()
        
        # Register hooks
        self._register_hooks()
        
        # Record initial memory
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        try:
            # Forward pass
            start_time = time.time()
            output = self.model(input_data)
            
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = time.time() - start_time
            
            # Compute loss
            loss = loss_fn(output, target)
            
            # Backward pass
            start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time = time.time() - start_time
            
            # Optimizer step
            start_time = time.time()
            optimizer.step()
            
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_time = time.time() - start_time
            
            # Classify tensors
            self._classify_tensors()
            
            # Print profiling summary
            print(f"\nTiming Summary:")
            print(f"  Forward:   {forward_time*1000:.2f} ms")
            print(f"  Backward:  {backward_time*1000:.2f} ms")
            print(f"  Optimizer: {optimizer_time*1000:.2f} ms")
            print(f"  Total:     {(forward_time + backward_time + optimizer_time)*1000:.2f} ms")
            
            if self.device == 'cuda' and torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"\nPeak GPU Memory: {peak_memory:.2f} MB")
            
        finally:
            # Clean up hooks
            self._remove_hooks()
        
        return self.graph
    
    def _classify_tensors(self):
        """Classify all tensors in the graph."""
        for tensor_id, tensor_info in self.graph.tensors.items():
            tensor_type = self.classifier.classify_tensor(
                tensor_info.shape,
                tensor_info.requires_grad,
                tensor_info.is_leaf
            )
            tensor_info.tensor_type = tensor_type
