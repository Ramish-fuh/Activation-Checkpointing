"""
Training and profiling script for ResNet-152
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.resnet152 import create_resnet152
from src.phase1_profiler import GraphProfiler, ActivationAnalyzer, MemoryVisualizer


def create_dummy_data(batch_size: int, num_batches: int = 10, 
                     image_size: int = 224, num_classes: int = 1000):
    """
    Create dummy ImageNet-like data for testing.
    
    Args:
        batch_size: Batch size
        num_batches: Number of batches to generate
        image_size: Image size (default 224 for ImageNet)
        num_classes: Number of classes
        
    Returns:
        DataLoader with dummy data
    """
    # Random images: (batch_size * num_batches, 3, 224, 224)
    images = torch.randn(batch_size * num_batches, 3, image_size, image_size)
    # Random labels
    labels = torch.randint(0, num_classes, (batch_size * num_batches,))
    
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def train_single_iteration(model, data, target, loss_fn, optimizer, device):
    """Run a single training iteration."""
    data, target = data.to(device), target.to(device)
    
    # Forward pass
    output = model(data)
    loss = loss_fn(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def profile_resnet(batch_size: int = 32, device: str = 'cpu', 
                  num_classes: int = 1000, pretrained: bool = False):
    """
    Profile ResNet-152 model.
    
    Args:
        batch_size: Batch size for profiling
        device: Device to use ('cpu', 'cuda', 'mps')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    """
    print(f"\n{'='*70}")
    print(f"Profiling ResNet-152")
    print(f"{'='*70}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"Pretrained: {pretrained}")
    print(f"{'='*70}\n")
    
    # Create model
    model = create_resnet152(num_classes=num_classes, pretrained=pretrained, device=device)
    model.print_model_info()
    
    # Create dummy data
    dataloader = create_dummy_data(batch_size=batch_size, num_batches=1, num_classes=num_classes)
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Create profiler
    profiler = GraphProfiler(model, device=device)
    
    # Profile one iteration
    print("Starting profiling...")
    graph = profiler.profile_iteration(data, target, loss_fn, optimizer)
    
    # Print graph summary
    graph.print_summary()
    
    # Analyze activations
    print("Analyzing activations...")
    analyzer = ActivationAnalyzer(graph)
    stats = analyzer.analyze()
    analyzer.print_summary(stats)
    
    # Generate visualizations
    print("Generating visualizations...")
    visualizer = MemoryVisualizer(output_dir="results/graphs/resnet152")
    visualizer.plot_memory_timeline(graph, filename=f"memory_timeline_bs{batch_size}.png")
    visualizer.plot_peak_memory_breakdown(graph, filename=f"peak_memory_breakdown_bs{batch_size}.png")
    
    print(f"\n{'='*70}")
    print(f"Profiling Complete!")
    print(f"Results saved to: results/graphs/resnet152/")
    print(f"{'='*70}\n")
    
    return graph, stats


def profile_multiple_batch_sizes(batch_sizes: list, device: str = 'cpu'):
    """
    Profile ResNet-152 with multiple batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        device: Device to use
    """
    peak_memories = []
    latencies = []
    
    for bs in batch_sizes:
        print(f"\n\nProfiling with batch size: {bs}")
        print(f"{'='*70}")
        
        try:
            # Create model
            model = create_resnet152(num_classes=1000, pretrained=False, device=device)
            
            # Create data
            dataloader = create_dummy_data(batch_size=bs, num_batches=1)
            data, target = next(iter(dataloader))
            data, target = data.to(device), target.to(device)
            
            # Profiler
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            profiler = GraphProfiler(model, device=device)
            
            # Time the iteration
            import time
            start = time.time()
            graph = profiler.profile_iteration(data, target, loss_fn, optimizer)
            elapsed = (time.time() - start) * 1000  # ms
            
            peak_mem = graph.compute_peak_memory() / (1024 ** 2)  # MB
            
            peak_memories.append(peak_mem)
            latencies.append(elapsed)
            
            print(f"Batch Size {bs}: Peak Memory = {peak_mem:.2f} MB, Latency = {elapsed:.2f} ms")
            
        except RuntimeError as e:
            print(f"Failed with batch size {bs}: {e}")
            break
    
    # Plot results
    if peak_memories:
        visualizer = MemoryVisualizer(output_dir="results/graphs/resnet152")
        visualizer.plot_memory_vs_batch_size(
            batch_sizes[:len(peak_memories)], 
            peak_memories,
            with_checkpointing=False,
            filename="memory_vs_batch_size_no_ac.png"
        )
        visualizer.plot_latency_vs_batch_size(
            batch_sizes[:len(latencies)],
            latencies,
            with_checkpointing=False,
            filename="latency_vs_batch_size_no_ac.png"
        )


def main():
    parser = argparse.ArgumentParser(description='Train and profile ResNet-152')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', 
                       choices=['cpu', 'cuda', 'mps'], help='Device to use')
    parser.add_argument('--profile', action='store_true', help='Run profiling')
    parser.add_argument('--multiple-batch-sizes', action='store_true',
                       help='Profile multiple batch sizes')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = 'cpu'
    
    if args.profile:
        if args.multiple_batch_sizes:
            batch_sizes = [8, 16, 32, 64, 128]
            profile_multiple_batch_sizes(batch_sizes, device=args.device)
        else:
            profile_resnet(batch_size=args.batch_size, device=args.device, 
                         pretrained=args.pretrained)
    else:
        # Simple training run without profiling
        print("Running simple training iteration...")
        model = create_resnet152(num_classes=1000, pretrained=args.pretrained, device=args.device)
        model.print_model_info()
        
        dataloader = create_dummy_data(batch_size=args.batch_size, num_batches=5)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        for i, (data, target) in enumerate(dataloader):
            loss = train_single_iteration(model, data, target, loss_fn, optimizer, args.device)
            print(f"Iteration {i+1}: Loss = {loss:.4f}")


if __name__ == '__main__':
    main()
