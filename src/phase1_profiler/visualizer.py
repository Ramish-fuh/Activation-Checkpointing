"""
Memory Visualizer: Generate graphs and plots for memory analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from .graph_builder import ComputationGraph


class MemoryVisualizer:
    """
    Creates visualizations of memory usage patterns during training.
    """
    
    def __init__(self, output_dir: str = "results/graphs"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_memory_timeline(self, graph: ComputationGraph, 
                            filename: str = "memory_timeline.png"):
        """
        Plot memory usage over time during the iteration.
        
        Args:
            graph: Computation graph with profiling data
            filename: Output filename
        """
        timeline = graph.get_memory_timeline()
        
        if not timeline:
            print("No timeline data available")
            return
        
        # Extract data
        operations = [node_id for node_id, _, _ in timeline]
        total_memory = [mem / (1024 ** 2) for _, mem, _ in timeline]  # Convert to MB
        
        # Extract memory by type
        memory_types = {}
        for _, _, breakdown in timeline:
            for mem_type in breakdown.keys():
                if mem_type not in memory_types:
                    memory_types[mem_type] = []
        
        for _, _, breakdown in timeline:
            for mem_type in memory_types.keys():
                memory_types[mem_type].append(breakdown.get(mem_type, 0) / (1024 ** 2))
        
        # Create stacked area plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Stack the areas
        bottom = np.zeros(len(operations))
        colors = {'parameter': '#1f77b4', 'gradient': '#ff7f0e', 
                 'activation': '#2ca02c', 'optimizer_state': '#d62728', 
                 'other': '#9467bd'}
        
        for mem_type, values in memory_types.items():
            ax.fill_between(operations, bottom, bottom + np.array(values),
                           label=mem_type.replace('_', ' ').title(),
                           color=colors.get(mem_type, '#gray'),
                           alpha=0.7)
            bottom += np.array(values)
        
        ax.set_xlabel('Operation Number', fontsize=12)
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title('Memory Usage Timeline by Tensor Type', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Memory timeline saved to: {output_path}")
    
    def plot_peak_memory_breakdown(self, graph: ComputationGraph,
                                   filename: str = "peak_memory_breakdown.png"):
        """
        Plot pie chart showing peak memory breakdown by tensor type.
        
        Args:
            graph: Computation graph
            filename: Output filename
        """
        # Get peak memory breakdown
        timeline = graph.get_memory_timeline()
        if not timeline:
            return
        
        # Find the point with maximum total memory
        max_idx = max(range(len(timeline)), key=lambda i: timeline[i][1])
        _, peak_memory, breakdown = timeline[max_idx]
        
        # Prepare data
        labels = []
        sizes = []
        colors_list = []
        colors = {'parameter': '#1f77b4', 'gradient': '#ff7f0e', 
                 'activation': '#2ca02c', 'optimizer_state': '#d62728', 
                 'other': '#9467bd'}
        
        for mem_type, size_bytes in breakdown.items():
            if size_bytes > 0:
                labels.append(mem_type.replace('_', ' ').title())
                sizes.append(size_bytes / (1024 ** 2))  # Convert to MB
                colors_list.append(colors.get(mem_type, '#gray'))
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list,
                                           autopct=lambda pct: f'{pct:.1f}%\n({pct*sum(sizes)/100:.1f} MB)',
                                           startangle=90)
        
        # Improve text
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        ax.set_title(f'Peak Memory Breakdown\nTotal: {sum(sizes):.2f} MB', 
                    fontsize=14, fontweight='bold')
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Peak memory breakdown saved to: {output_path}")
    
    def plot_memory_vs_batch_size(self, batch_sizes: List[int], 
                                  peak_memories: List[float],
                                  with_checkpointing: bool = False,
                                  filename: str = "memory_vs_batch_size.png"):
        """
        Plot peak memory consumption vs batch size.
        
        Args:
            batch_sizes: List of batch sizes tested
            peak_memories: Corresponding peak memories (in MB)
            with_checkpointing: Whether this includes activation checkpointing
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        label = "With Activation Checkpointing" if with_checkpointing else "Without Activation Checkpointing"
        color = '#2ca02c' if with_checkpointing else '#d62728'
        marker = 's' if with_checkpointing else 'o'
        
        ax.plot(batch_sizes, peak_memories, marker=marker, linewidth=2, 
               markersize=8, label=label, color=color)
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax.set_title('Peak Memory Consumption vs Batch Size', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Memory vs batch size plot saved to: {output_path}")
    
    def plot_latency_vs_batch_size(self, batch_sizes: List[int],
                                   latencies: List[float],
                                   with_checkpointing: bool = False,
                                   filename: str = "latency_vs_batch_size.png"):
        """
        Plot iteration latency vs batch size.
        
        Args:
            batch_sizes: List of batch sizes tested
            latencies: Corresponding latencies (in ms)
            with_checkpointing: Whether this includes activation checkpointing
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        label = "With Activation Checkpointing" if with_checkpointing else "Without Activation Checkpointing"
        color = '#2ca02c' if with_checkpointing else '#1f77b4'
        marker = 's' if with_checkpointing else 'o'
        
        ax.plot(batch_sizes, latencies, marker=marker, linewidth=2,
               markersize=8, label=label, color=color)
        
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Iteration Latency (ms)', fontsize=12)
        ax.set_title('Iteration Latency vs Batch Size', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Latency vs batch size plot saved to: {output_path}")
