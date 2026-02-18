"""
Graph Builder: Constructs computational graph from PyTorch execution.
"""

import torch
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class GraphNode:
    """Represents a single operation in the computation graph."""
    
    node_id: int
    operation_name: str
    operation_type: str
    
    # Input/Output tensors
    input_tensors: List[int] = field(default_factory=list)  # tensor IDs
    output_tensors: List[int] = field(default_factory=list)  # tensor IDs
    
    # Profiling data
    forward_time: float = 0.0  # milliseconds
    backward_time: float = 0.0  # milliseconds
    memory_allocated: int = 0  # bytes
    
    # Classification
    is_forward: bool = True
    is_backward: bool = False
    is_optimizer: bool = False
    
    def __repr__(self):
        return f"GraphNode(id={self.node_id}, op={self.operation_name}, type={self.operation_type})"


@dataclass
class TensorInfo:
    """Metadata about a tensor in the computation graph."""
    
    tensor_id: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int
    
    # Lifecycle tracking
    producer_node: Optional[int] = None  # Node that creates this tensor
    consumer_nodes: List[int] = field(default_factory=list)  # Nodes that use this tensor
    first_use: Optional[int] = None  # First node that uses it
    last_use: Optional[int] = None  # Last node that uses it
    
    # Classification
    tensor_type: str = "unknown"  # parameter, gradient, activation, optimizer_state, other
    requires_grad: bool = False
    is_leaf: bool = False
    
    def __repr__(self):
        return f"TensorInfo(id={self.tensor_id}, shape={self.shape}, type={self.tensor_type})"


class ComputationGraph:
    """
    Represents the full computation graph for one training iteration.
    Includes forward pass, backward pass, and optimizer step.
    """
    
    def __init__(self):
        self.nodes: Dict[int, GraphNode] = {}
        self.tensors: Dict[int, TensorInfo] = {}
        
        self.node_counter: int = 0
        self.tensor_counter: int = 0
        
        # Track execution order
        self.execution_order: List[int] = []  # List of node IDs in execution order
        
        # For building graph during execution
        self.tensor_to_id: Dict[int, int] = {}  # Maps Python id(tensor) to our tensor_id
        
    def add_node(self, operation_name: str, operation_type: str, 
                 is_forward: bool = True, is_backward: bool = False) -> GraphNode:
        """Add a new operation node to the graph."""
        node = GraphNode(
            node_id=self.node_counter,
            operation_name=operation_name,
            operation_type=operation_type,
            is_forward=is_forward,
            is_backward=is_backward
        )
        self.nodes[self.node_counter] = node
        self.execution_order.append(self.node_counter)
        self.node_counter += 1
        return node
    
    def add_tensor(self, tensor: torch.Tensor) -> int:
        """Register a tensor and return its ID."""
        python_id = id(tensor)
        
        if python_id in self.tensor_to_id:
            return self.tensor_to_id[python_id]
        
        tensor_id = self.tensor_counter
        self.tensor_to_id[python_id] = tensor_id
        
        tensor_info = TensorInfo(
            tensor_id=tensor_id,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            size_bytes=tensor.element_size() * tensor.numel(),
            requires_grad=tensor.requires_grad,
            is_leaf=tensor.is_leaf
        )
        
        self.tensors[tensor_id] = tensor_info
        self.tensor_counter += 1
        return tensor_id
    
    def add_edge(self, from_node: int, to_node: int, tensor_id: int, is_input: bool = True):
        """Add an edge representing data flow between nodes."""
        if is_input:
            self.nodes[to_node].input_tensors.append(tensor_id)
            if self.tensors[tensor_id].first_use is None:
                self.tensors[tensor_id].first_use = to_node
            self.tensors[tensor_id].consumer_nodes.append(to_node)
            self.tensors[tensor_id].last_use = to_node
        else:
            self.nodes[from_node].output_tensors.append(tensor_id)
            self.tensors[tensor_id].producer_node = from_node
    
    def get_topological_order(self) -> List[int]:
        """Return nodes in topological order."""
        # For now, return execution order (which should be topological)
        return self.execution_order
    
    def get_activations(self) -> List[int]:
        """Get all activation tensor IDs."""
        return [tid for tid, tinfo in self.tensors.items() 
                if tinfo.tensor_type == "activation"]
    
    def get_parameters(self) -> List[int]:
        """Get all parameter tensor IDs."""
        return [tid for tid, tinfo in self.tensors.items() 
                if tinfo.tensor_type == "parameter"]
    
    def compute_peak_memory(self) -> int:
        """Compute peak memory usage across all operations."""
        current_memory = 0
        peak_memory = 0
        
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            
            # Add memory for outputs
            for tensor_id in node.output_tensors:
                current_memory += self.tensors[tensor_id].size_bytes
            
            # Check for peak
            peak_memory = max(peak_memory, current_memory)
            
            # Free memory for inputs if this is their last use
            for tensor_id in node.input_tensors:
                if self.tensors[tensor_id].last_use == node_id:
                    current_memory -= self.tensors[tensor_id].size_bytes
        
        return peak_memory
    
    def get_memory_timeline(self) -> List[Tuple[int, int, Dict[str, int]]]:
        """
        Get memory usage at each operation.
        Returns: List of (node_id, total_memory, breakdown_by_type)
        """
        timeline = []
        active_tensors = set()
        
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            
            # Add newly created tensors
            for tensor_id in node.output_tensors:
                active_tensors.add(tensor_id)
            
            # Calculate current memory by type
            memory_by_type = defaultdict(int)
            for tensor_id in active_tensors:
                tinfo = self.tensors[tensor_id]
                memory_by_type[tinfo.tensor_type] += tinfo.size_bytes
            
            total_memory = sum(memory_by_type.values())
            timeline.append((node_id, total_memory, dict(memory_by_type)))
            
            # Remove tensors that are no longer needed
            to_remove = []
            for tensor_id in active_tensors:
                if self.tensors[tensor_id].last_use == node_id:
                    to_remove.append(tensor_id)
            for tensor_id in to_remove:
                active_tensors.remove(tensor_id)
        
        return timeline
    
    def print_summary(self):
        """Print a summary of the computation graph."""
        print(f"\n{'='*60}")
        print(f"Computation Graph Summary")
        print(f"{'='*60}")
        print(f"Total Nodes: {len(self.nodes)}")
        print(f"Total Tensors: {len(self.tensors)}")
        
        # Count by type
        tensor_types = defaultdict(int)
        tensor_memory = defaultdict(int)
        for tinfo in self.tensors.values():
            tensor_types[tinfo.tensor_type] += 1
            tensor_memory[tinfo.tensor_type] += tinfo.size_bytes
        
        print(f"\nTensor Breakdown:")
        for ttype, count in sorted(tensor_types.items()):
            memory_mb = tensor_memory[ttype] / (1024 ** 2)
            print(f"  {ttype:15s}: {count:4d} tensors, {memory_mb:8.2f} MB")
        
        peak_memory = self.compute_peak_memory()
        print(f"\nPeak Memory: {peak_memory / (1024 ** 2):.2f} MB")
        print(f"{'='*60}\n")
