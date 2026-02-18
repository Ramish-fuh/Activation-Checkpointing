"""
Training and profiling script for BERT
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

from src.models.bert import create_bert, get_bert_tokenizer
from src.phase1_profiler import GraphProfiler, ActivationAnalyzer, MemoryVisualizer


def create_dummy_data(batch_size: int, seq_length: int = 128, 
                     num_batches: int = 10, num_labels: int = 2):
    """
    Create dummy BERT input data for testing.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        num_batches: Number of batches
        num_labels: Number of labels for classification
        
    Returns:
        DataLoader with dummy data
    """
    # Random token IDs (assuming vocab size ~30000 for BERT)
    input_ids = torch.randint(0, 30000, (batch_size * num_batches, seq_length))
    # Attention masks (all ones for simplicity)
    attention_mask = torch.ones(batch_size * num_batches, seq_length, dtype=torch.long)
    # Random labels
    labels = torch.randint(0, num_labels, (batch_size * num_batches,))
    
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def train_single_iteration(model, input_ids, attention_mask, labels, 
                          loss_fn, optimizer, device):
    """Run a single training iteration."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    loss = loss_fn(logits, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def profile_bert(batch_size: int = 16, seq_length: int = 128, 
                device: str = 'cpu', model_name: str = 'bert-base-uncased',
                num_labels: int = 2, pretrained: bool = False):
    """
    Profile BERT model.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        device: Device to use
        model_name: BERT model variant
        num_labels: Number of labels
        pretrained: Whether to use pretrained weights
    """
    print(f"\n{'='*70}")
    print(f"Profiling BERT")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {seq_length}")
    print(f"Device: {device}")
    print(f"Pretrained: {pretrained}")
    print(f"{'='*70}\n")
    
    # Create model
    model = create_bert(model_name=model_name, num_labels=num_labels, 
                       pretrained=pretrained, device=device)
    model.print_model_info()
    
    # Create dummy data
    dataloader = create_dummy_data(batch_size=batch_size, seq_length=seq_length, 
                                  num_batches=1, num_labels=num_labels)
    input_ids, attention_mask, labels = next(iter(dataloader))
    
    # For BERT, we need to package inputs differently for the profiler
    # Create a wrapper that accepts the packed inputs
    class BERTProfileWrapper(nn.Module):
        def __init__(self, bert_model):
            super().__init__()
            self.bert_model = bert_model
            
        def forward(self, x):
            # x is a tuple of (input_ids, attention_mask)
            input_ids, attention_mask = x[:, 0, :].long(), x[:, 1, :].long()
            return self.bert_model(input_ids, attention_mask)
    
    # Package inputs
    # Stack input_ids and attention_mask along dim 1
    packed_input = torch.stack([input_ids.float(), attention_mask.float()], dim=1)
    packed_input = packed_input.to(device)
    labels = labels.to(device)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    # Simple profiling without the full profiler for now
    print("Running training iteration...")
    
    import time
    start = time.time()
    
    # Forward
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    output = model(input_ids, attention_mask)
    loss = loss_fn(output, labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    elapsed = (time.time() - start) * 1000
    
    print(f"\nIteration completed in {elapsed:.2f} ms")
    print(f"Loss: {loss.item():.4f}")
    
    if device == 'cuda' and torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU Memory: {peak_memory:.2f} MB")
    
    print(f"\n{'='*70}")
    print(f"Basic profiling complete!")
    print(f"Note: Full graph profiling for BERT coming soon")
    print(f"{'='*70}\n")


def profile_multiple_batch_sizes(batch_sizes: list, seq_length: int = 128, 
                                 device: str = 'cpu'):
    """
    Profile BERT with multiple batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test
        seq_length: Sequence length
        device: Device to use
    """
    peak_memories = []
    latencies = []
    
    for bs in batch_sizes:
        print(f"\n\nProfiling with batch size: {bs}")
        print(f"{'='*70}")
        
        try:
            # Create model
            model = create_bert(model_name='bert-base-uncased', num_labels=2, 
                              pretrained=False, device=device)
            
            # Create data
            dataloader = create_dummy_data(batch_size=bs, seq_length=seq_length, num_batches=1)
            input_ids, attention_mask, labels = next(iter(dataloader))
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=2e-5)
            
            # Reset memory stats
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Time the iteration
            import time
            start = time.time()
            
            output = model(input_ids, attention_mask)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start) * 1000  # ms
            
            if device == 'cuda' and torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_mem = 0
            
            peak_memories.append(peak_mem)
            latencies.append(elapsed)
            
            print(f"Batch Size {bs}: Peak Memory = {peak_mem:.2f} MB, Latency = {elapsed:.2f} ms")
            
        except RuntimeError as e:
            print(f"Failed with batch size {bs}: {e}")
            break
    
    # Plot results
    if peak_memories and device == 'cuda':
        visualizer = MemoryVisualizer(output_dir="results/graphs/bert")
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
    parser = argparse.ArgumentParser(description='Train and profile BERT')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=128, help='Sequence length')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'], help='Device to use')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                       help='BERT model variant')
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
            batch_sizes = [4, 8, 16, 32, 64]
            profile_multiple_batch_sizes(batch_sizes, seq_length=args.seq_length, 
                                       device=args.device)
        else:
            profile_bert(batch_size=args.batch_size, seq_length=args.seq_length,
                       device=args.device, model_name=args.model_name,
                       pretrained=args.pretrained)
    else:
        # Simple training run
        print("Running simple training iteration...")
        model = create_bert(model_name=args.model_name, num_labels=2, 
                          pretrained=args.pretrained, device=args.device)
        model.print_model_info()
        
        dataloader = create_dummy_data(batch_size=args.batch_size, 
                                      seq_length=args.seq_length, num_batches=5)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        
        for i, (input_ids, attention_mask, labels) in enumerate(dataloader):
            loss = train_single_iteration(model, input_ids, attention_mask, labels,
                                        loss_fn, optimizer, args.device)
            print(f"Iteration {i+1}: Loss = {loss:.4f}")


if __name__ == '__main__':
    main()
