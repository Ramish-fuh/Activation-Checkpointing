"""
Activation Analyzer: Analyzes activation lifetimes and memory patterns.
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .graph_builder import ComputationGraph, TensorInfo


@dataclass
class ActivationLifetime:
    """Lifetime analysis for an activation tensor."""
    
    tensor_id: int
    shape: Tuple[int, ...]
    size_bytes: int
    
    # Lifecycle
    creation_node: int
    first_use_node: int
    last_use_node: int
    
    # Metrics
    lifetime_ops: int  # Number of operations it lives through
    idle_time_percent: float  # % of lifetime spent idle
    
    def __repr__(self):
        return (f"ActivationLifetime(id={self.tensor_id}, "
                f"size={self.size_bytes / (1024**2):.2f}MB, "
                f"lifetime={self.lifetime_ops} ops)")


class ActivationAnalyzer:
    """
    Analyzes activation tensors to understand memory usage patterns
    and identify optimization opportunities.
    """
    
    def __init__(self, graph: ComputationGraph):
        """
        Initialize analyzer with a computation graph.
        
        Args:
            graph: Computation graph to analyze
        """
        self.graph = graph
        self.activation_lifetimes: Dict[int, ActivationLifetime] = {}
    
    def analyze(self) -> Dict[str, any]:
        """
        Perform complete activation analysis.
        
        Returns:
            Dictionary with analysis results
        """
        # Get all activation tensors
        activation_ids = self.graph.get_activations()
        
        # Analyze each activation
        for tensor_id in activation_ids:
            lifetime = self._analyze_activation(tensor_id)
            if lifetime:
                self.activation_lifetimes[tensor_id] = lifetime
        
        # Compute statistics
        stats = self._compute_statistics()
        
        return stats
    
    def _analyze_activation(self, tensor_id: int) -> Optional[ActivationLifetime]:
        """Analyze lifetime of a single activation."""
        tensor_info = self.graph.tensors.get(tensor_id)
        if not tensor_info or tensor_info.tensor_type != "activation":
            return None
        
        creation_node = tensor_info.producer_node
        first_use = tensor_info.first_use
        last_use = tensor_info.last_use
        
        if creation_node is None or last_use is None:
            return None
        
        # Calculate lifetime in terms of operations
        exec_order = self.graph.execution_order
        try:
            creation_idx = exec_order.index(creation_node)
            last_use_idx = exec_order.index(last_use)
            lifetime_ops = last_use_idx - creation_idx
        except ValueError:
            return None
        
        # Calculate idle time (simplified)
        # In reality, we'd track actual usage vs. existence time
        num_uses = len(tensor_info.consumer_nodes)
        idle_time_percent = max(0, (lifetime_ops - num_uses) / max(lifetime_ops, 1) * 100)
        
        return ActivationLifetime(
            tensor_id=tensor_id,
            shape=tensor_info.shape,
            size_bytes=tensor_info.size_bytes,
            creation_node=creation_node,
            first_use_node=first_use if first_use else creation_node,
            last_use_node=last_use,
            lifetime_ops=lifetime_ops,
            idle_time_percent=idle_time_percent
        )
    
    def _compute_statistics(self) -> Dict[str, any]:
        """Compute aggregate statistics on activations."""
        if not self.activation_lifetimes:
            return {}
        
        lifetimes = list(self.activation_lifetimes.values())
        
        # Total activation memory
        total_activation_memory = sum(lt.size_bytes for lt in lifetimes)
        
        # Peak activation memory (at any point in time)
        peak_activation_memory = self._compute_peak_activation_memory()
        
        # Average lifetime
        avg_lifetime = sum(lt.lifetime_ops for lt in lifetimes) / len(lifetimes)
        
        # Long-lived activations (lifetime > 50% of total ops)
        total_ops = len(self.graph.execution_order)
        long_lived = [lt for lt in lifetimes if lt.lifetime_ops > total_ops * 0.5]
        
        # Sort by memory consumption
        by_memory = sorted(lifetimes, key=lambda x: x.size_bytes, reverse=True)
        top_10_memory = by_memory[:min(10, len(by_memory))]
        
        # Sort by lifetime
        by_lifetime = sorted(lifetimes, key=lambda x: x.lifetime_ops, reverse=True)
        top_10_lifetime = by_lifetime[:min(10, len(by_lifetime))]
        
        stats = {
            'total_activations': len(lifetimes),
            'total_activation_memory_bytes': total_activation_memory,
            'total_activation_memory_mb': total_activation_memory / (1024 ** 2),
            'peak_activation_memory_bytes': peak_activation_memory,
            'peak_activation_memory_mb': peak_activation_memory / (1024 ** 2),
            'avg_lifetime_ops': avg_lifetime,
            'num_long_lived': len(long_lived),
            'top_10_by_memory': top_10_memory,
            'top_10_by_lifetime': top_10_lifetime
        }
        
        return stats
    
    def _compute_peak_activation_memory(self) -> int:
        """Compute peak memory used by activations at any point."""
        active_activations = set()
        peak_memory = 0
        
        for node_id in self.graph.execution_order:
            node = self.graph.nodes[node_id]
            
            # Add newly created activations
            for tensor_id in node.output_tensors:
                if self.graph.tensors[tensor_id].tensor_type == "activation":
                    active_activations.add(tensor_id)
            
            # Compute current memory
            current_memory = sum(
                self.graph.tensors[tid].size_bytes 
                for tid in active_activations
            )
            peak_memory = max(peak_memory, current_memory)
            
            # Remove activations that are no longer needed
            to_remove = []
            for tensor_id in active_activations:
                if self.graph.tensors[tensor_id].last_use == node_id:
                    to_remove.append(tensor_id)
            for tensor_id in to_remove:
                active_activations.remove(tensor_id)
        
        return peak_memory
    
    def print_summary(self, stats: Dict[str, any]):
        """Print a summary of the activation analysis."""
        print(f"\n{'='*60}")
        print(f"Activation Analysis Summary")
        print(f"{'='*60}")
        print(f"Total Activations: {stats['total_activations']}")
        print(f"Total Activation Memory: {stats['total_activation_memory_mb']:.2f} MB")
        print(f"Peak Activation Memory: {stats['peak_activation_memory_mb']:.2f} MB")
        print(f"Average Lifetime: {stats['avg_lifetime_ops']:.1f} operations")
        print(f"Long-lived Activations: {stats['num_long_lived']}")
        
        print(f"\nTop 10 Activations by Memory:")
        for i, lt in enumerate(stats['top_10_by_memory'], 1):
            print(f"  {i}. Tensor {lt.tensor_id}: {lt.size_bytes / (1024**2):.2f} MB, "
                  f"lifetime={lt.lifetime_ops} ops")
        
        print(f"\nTop 10 Activations by Lifetime:")
        for i, lt in enumerate(stats['top_10_by_lifetime'], 1):
            print(f"  {i}. Tensor {lt.tensor_id}: {lt.lifetime_ops} ops, "
                  f"size={lt.size_bytes / (1024**2):.2f} MB")
        
        print(f"{'='*60}\n")
