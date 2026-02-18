"""
Memory tracking utilities.
"""

import torch
import psutil
import os
from typing import Dict


def get_memory_stats(device: str = 'cpu') -> Dict[str, float]:
    """
    Get current memory statistics.
    
    Args:
        device: Device to check ('cpu', 'cuda', 'mps')
        
    Returns:
        Dictionary with memory stats in MB
    """
    stats = {}
    
    if device == 'cuda' and torch.cuda.is_available():
        stats['cuda_allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats['cuda_reserved'] = torch.cuda.memory_reserved() / (1024 ** 2)
        stats['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        stats['cuda_max_reserved'] = torch.cuda.max_memory_reserved() / (1024 ** 2)
    
    # CPU memory
    process = psutil.Process(os.getpid())
    stats['cpu_rss'] = process.memory_info().rss / (1024 ** 2)
    stats['cpu_vms'] = process.memory_info().vms / (1024 ** 2)
    
    return stats


def reset_memory_stats(device: str = 'cpu'):
    """Reset memory statistics."""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def print_memory_stats(device: str = 'cpu'):
    """Print current memory statistics."""
    stats = get_memory_stats(device)
    
    print(f"\nMemory Statistics ({device}):")
    print(f"{'='*50}")
    
    if device == 'cuda':
        print(f"CUDA Allocated:     {stats.get('cuda_allocated', 0):.2f} MB")
        print(f"CUDA Reserved:      {stats.get('cuda_reserved', 0):.2f} MB")
        print(f"CUDA Max Allocated: {stats.get('cuda_max_allocated', 0):.2f} MB")
        print(f"CUDA Max Reserved:  {stats.get('cuda_max_reserved', 0):.2f} MB")
    
    print(f"CPU RSS:            {stats['cpu_rss']:.2f} MB")
    print(f"CPU VMS:            {stats['cpu_vms']:.2f} MB")
    print(f"{'='*50}\n")


def get_tensor_memory(tensor: torch.Tensor) -> int:
    """
    Get memory size of a tensor in bytes.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Memory size in bytes
    """
    return tensor.element_size() * tensor.numel()


def format_bytes(bytes_val: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_val: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"
