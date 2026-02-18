"""
Main Graph Profiler: Orchestrates graph capture and profiling.
"""

import torch
import time
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from torch.profiler import profile, ProfilerActivity, record_function

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
        
        # Track parameters for gradient tracking
        self.param_to_tensor_id = {}
        self.backward_start_node = None
        
        # Track intermediate activations for gradient capture
        self.intermediate_tensors = {}  # tensor_id -> tensor reference
        self.tensor_to_node = {}  # tensor_id -> node_id that produced it
        
        # Backward timing data
        self.backward_timings = {}  # operation_name -> time_ms
        
        # Track current operation
        self.current_node: Optional[GraphNode] = None
        self.operation_counter = 0
        
        # Memory tracking
        self.memory_snapshots = []
        
    def _register_parameters(self):
        """Register model parameters before profiling."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                tensor_id = self.graph.add_tensor(param, tensor_type="parameter")
                self.param_to_tensor_id[id(param)] = tensor_id
    
    def _register_hooks(self):
        """Register forward hooks on all modules."""
        
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
            
            # Record outputs and retain gradients for intermediate tracking
            if isinstance(outputs, torch.Tensor):
                tensor_id = self.graph.add_tensor(outputs)
                self.graph.add_edge(node.node_id, None, tensor_id, is_input=False)
                # Keep reference and retain gradient for intermediate tensors
                if outputs.requires_grad and not outputs.is_leaf:
                    self.intermediate_tensors[tensor_id] = outputs
                    self.tensor_to_node[tensor_id] = node.node_id
                    outputs.retain_grad()
            elif isinstance(outputs, tuple):
                for out in outputs:
                    if isinstance(out, torch.Tensor):
                        tensor_id = self.graph.add_tensor(out)
                        self.graph.add_edge(node.node_id, None, tensor_id, is_input=False)
                        # Keep reference and retain gradient for intermediate tensors
                        if out.requires_grad and not out.is_leaf:
                            self.intermediate_tensors[tensor_id] = out
                            self.tensor_to_node[tensor_id] = node.node_id
                            out.retain_grad()
            
            # Memory snapshot
            if self.device == 'cuda' and torch.cuda.is_available():
                node.memory_allocated = torch.cuda.memory_allocated()
        
        # Register forward hooks only on leaf modules
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                h = module.register_forward_hook(forward_hook)
                self.hooks.append(h)
    
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
            
            # Backward pass with detailed profiling
            start_time = time.time()
            optimizer.zero_grad()
            
            # Use PyTorch profiler to capture per-operation backward timing
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=False,
                with_stack=False
            ) as prof:
                loss.backward()
            
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time = time.time() - start_time
            
            # Extract per-operation backward timing from profiler
            self._extract_backward_timings(prof)
            
            # Track intermediate gradients
            self._track_intermediate_gradients()
            
            # Track parameter gradients (linked to forward nodes)
            self._track_parameter_gradients()
            
            # Optimizer step
            start_time = time.time()
            optimizer.step()
            
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_time = time.time() - start_time
            
            # Track optimizer state
            self._track_optimizer_state(optimizer)
            
            # Classify tensors
            self._classify_tensors()
            
            # Mark backward nodes
            self._mark_backward_nodes()
            
            # Print profiling summary
            print(f"\nTiming Summary:")
            print(f"  Forward:   {forward_time*1000:.2f} ms")
            print(f"  Backward:  {backward_time*1000:.2f} ms")
            print(f"  Optimizer: {optimizer_time*1000:.2f} ms")
            print(f"  Total:     {(forward_time + backward_time + optimizer_time)*1000:.2f} ms")
            
            # Print complete backward operation timing breakdown
            if self.backward_timings:
                print(f"\n" + "="*70)
                print(f"Complete Per-Operation Backward Timing:")
                print("="*70)
                sorted_timings = sorted(self.backward_timings.items(), key=lambda x: x[1], reverse=True)
                for idx, (op_name, op_time) in enumerate(sorted_timings, 1):
                    print(f"  {idx}. {op_name}: {op_time:.2f} ms")
            
            # Print detailed operation list in topological order (Requirement 1)
            print(f"\n" + "="*70)
            print(f"Operations in Topological Order (Requirement 1):")
            print("="*70)
            topo_order = self.graph.compute_topological_order()
            print(f"Total Operations: {len(topo_order)}")
            print(f"\nShowing all {len(topo_order)} operations:")
            for idx, node_id in enumerate(topo_order, 1):
                node = self.graph.nodes[node_id]
                phase = "Forward" if node.is_forward else ("Backward" if node.is_backward else "Optimizer")
                time_info = f"{node.backward_time:.2f}ms" if node.backward_time > 0 else "N/A"
                mem_info = f"{node.memory_allocated/(1024**2):.2f}MB" if node.memory_allocated > 0 else "N/A"
                print(f"  {idx}. [{phase}] {node.operation_name} ({node.operation_type})")
                print(f"      Time: {time_info}, Memory: {mem_info}")
            
            # Print complete tensor categorization (Requirement 2)
            print(f"\n" + "="*70)
            print(f"Complete Tensor Categorization (Requirement 2):")
            print("="*70)
            tensor_by_type = {'parameter': [], 'gradient': [], 'activation': [], 'optimizer_state': [], 'other': []}
            for tensor_id, tensor_info in self.graph.tensors.items():
                tensor_by_type[tensor_info.tensor_type].append((tensor_id, tensor_info))
            
            for tensor_type, tensors in tensor_by_type.items():
                print(f"\n{tensor_type.upper()} Tensors: {len(tensors)}")
                print(f"Showing all {len(tensors)} {tensor_type} tensors:")
                for idx, (tid, tinfo) in enumerate(tensors, 1):
                    print(f"  {idx}. Tensor {tid}: shape={tinfo.shape}, size={tinfo.size_bytes/(1024**2):.2f}MB")
            
            # Print complete activation analysis with first/last use (Requirement 3)
            print(f"\n" + "="*70)
            print(f"Complete Activation Analysis - First and Last Use (Requirement 3):")
            print("="*70)
            activations = [(tid, tinfo) for tid, tinfo in self.graph.tensors.items() 
                          if tinfo.tensor_type == "activation"]
            print(f"Total Activations: {len(activations)}")
            print(f"\nShowing all {len(activations)} activations:")
            for idx, (tensor_id, tensor_info) in enumerate(activations, 1):
                first_use_node = self.graph.nodes.get(tensor_info.first_use)
                last_use_node = self.graph.nodes.get(tensor_info.last_use)
                first_use_name = first_use_node.operation_name if first_use_node else "Unknown"
                last_use_name = last_use_node.operation_name if last_use_node else "Unknown"
                lifetime = tensor_info.last_use - tensor_info.first_use if tensor_info.first_use and tensor_info.last_use else 0
                
                print(f"  {idx}. Activation Tensor {tensor_id}:")
                print(f"      Shape: {tensor_info.shape}, Size: {tensor_info.size_bytes/(1024**2):.2f}MB")
                print(f"      First Use: Node {tensor_info.first_use} ({first_use_name})")
                print(f"      Last Use: Node {tensor_info.last_use} ({last_use_name})")
                print(f"      Lifetime: {lifetime} operations")
            
            if self.device == 'cuda' and torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"\nPeak GPU Memory: {peak_memory:.2f} MB")
            
        finally:
            # Clean up hooks
            self._remove_hooks()
        
        return self.graph
    
    def _extract_backward_timings(self, prof):
        """Extract backward operation timings from profiler."""
        for event in prof.key_averages():
            event_name = event.key
            # Filter backward operations
            if 'Backward' in event_name or 'backward' in event_name:
                self.backward_timings[event_name] = event.cpu_time_total / 1000.0  # Convert to ms
    
    def _track_intermediate_gradients(self):
        """Track intermediate gradient tensors created during backward pass."""
        for tensor_id, tensor in self.intermediate_tensors.items():
            if tensor.grad is not None:
                # Add gradient tensor
                grad_tensor_id = self.graph.add_tensor(tensor.grad)
                grad_info = self.graph.tensors[grad_tensor_id]
                grad_info.tensor_type = "gradient"
                
                # Link gradient to the forward operation that created this activation
                if tensor_id in self.tensor_to_node:
                    forward_node_id = self.tensor_to_node[tensor_id]
                    forward_node = self.graph.nodes[forward_node_id]
                    
                    # Create corresponding backward node
                    backward_node = self.graph.add_node(
                        operation_name=f"backward_{forward_node.operation_name}",
                        operation_type=f"{forward_node.operation_type}Backward",
                        is_forward=False,
                        is_backward=True
                    )
                    
                    # Link activation tensor to backward node as input
                    backward_node.input_tensors.append(tensor_id)
                    
                    # Link gradient tensor as output of backward node
                    backward_node.output_tensors.append(grad_tensor_id)
                    self.graph.tensors[grad_tensor_id].producer_node = backward_node.node_id
                    
                    # Set backward timing if available
                    backward_op_name = f"{forward_node.operation_type}Backward"
                    if backward_op_name in self.backward_timings:
                        backward_node.backward_time = self.backward_timings[backward_op_name]
    
    def _track_parameter_gradients(self):
        """Track parameter gradient tensors and link to forward operations."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Add gradient tensor
                grad_tensor_id = self.graph.add_tensor(param.grad)
                grad_info = self.graph.tensors[grad_tensor_id]
                grad_info.tensor_type = "gradient"
                
                # Find the forward node that uses this parameter
                param_tensor_id = id(param)
                corresponding_forward_node = None
                
                for node_id, node in self.graph.nodes.items():
                    if node.is_forward:
                        for input_tid in node.input_tensors:
                            tensor_info = self.graph.tensors.get(input_tid)
                            if tensor_info and tensor_info.tensor_type == "parameter":
                                # Match parameter by shape
                                if tensor_info.shape == tuple(param.shape):
                                    corresponding_forward_node = node
                                    break
                    if corresponding_forward_node:
                        break
                
                # Create backward node linked to forward operation
                if corresponding_forward_node:
                    backward_node = self.graph.add_node(
                        operation_name=f"backward_{corresponding_forward_node.operation_name}_param",
                        operation_type=f"{corresponding_forward_node.operation_type}ParamGrad",
                        is_forward=False,
                        is_backward=True
                    )
                    backward_node.output_tensors.append(grad_tensor_id)
                    self.graph.tensors[grad_tensor_id].producer_node = backward_node.node_id
                    
                    # Set timing if available
                    backward_op_name = f"{corresponding_forward_node.operation_type}Backward"
                    if backward_op_name in self.backward_timings:
                        backward_node.backward_time = self.backward_timings[backward_op_name]
                else:
                    # Fallback: create standalone gradient node
                    backward_node = self.graph.add_node(
                        operation_name=f"gradient_{name}",
                        operation_type="GradientComputation",
                        is_forward=False,
                        is_backward=True
                    )
                    backward_node.output_tensors.append(grad_tensor_id)
                    self.graph.tensors[grad_tensor_id].producer_node = backward_node.node_id
    
    def _track_gradients(self):
        """Track gradient tensors after backward pass (legacy method)."""
        # This method is now replaced by _track_parameter_gradients and _track_intermediate_gradients
        pass
    
    def _track_optimizer_state(self, optimizer):
        """Track optimizer state tensors after optimizer step."""
        for group_idx, group in enumerate(optimizer.param_groups):
            for param_idx, param in enumerate(group['params']):
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for state_key, state_value in state.items():
                        if isinstance(state_value, torch.Tensor):
                            # Add optimizer state tensor
                            state_tensor_id = self.graph.add_tensor(state_value)
                            state_info = self.graph.tensors[state_tensor_id]
                            state_info.tensor_type = "optimizer_state"
                            
                            # Create node for optimizer state update
                            node = self.graph.add_node(
                                operation_name=f"optimizer_state_{group_idx}_{param_idx}_{state_key}",
                                operation_type="OptimizerState",
                                is_forward=False,
                                is_backward=False
                            )
                            node.output_tensors.append(state_tensor_id)
    
    def _mark_backward_nodes(self):
        """Mark nodes created during backward pass based on their type."""
        # Backward nodes are already marked during creation in _track_intermediate_gradients
        # and _track_parameter_gradients, so this is now a no-op
        pass
    
    def _classify_tensors(self):
        """Classify all tensors in the graph."""
        for tensor_id, tensor_info in self.graph.tensors.items():
            # Skip if already classified
            if tensor_info.tensor_type != "unknown":
                continue
                
            tensor_type = self.classifier.classify_tensor(
                tensor_info.shape,
                tensor_info.requires_grad,
                tensor_info.is_leaf
            )
            tensor_info.tensor_type = tensor_type
