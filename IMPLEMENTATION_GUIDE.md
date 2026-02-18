# CS265 Systems Project - Implementation Summary

## Project Structure Created

### Core Implementation (Phase 1 - Graph Profiler)

**Phase 1 Components (35%):**
- `src/phase1_profiler/graph_builder.py` - Computational graph data structures
- `src/phase1_profiler/profiler.py` - Main profiling orchestrator
- `src/phase1_profiler/tensor_classifier.py` - Tensor classification logic
- `src/phase1_profiler/activation_analyzer.py` - Activation lifetime analysis
- `src/phase1_profiler/visualizer.py` - Memory visualization and plotting

**Models:**
- `src/models/resnet152.py` - ResNet-152 wrapper
- `src/models/bert.py` - BERT wrapper

**Experiments:**
- `experiments/train_resnet.py` - ResNet-152 training and profiling
- `experiments/train_bert.py` - BERT training and profiling

**Utilities:**
- `src/utils/memory_utils.py` - Memory tracking functions
- `src/utils/graph_utils.py` - Graph manipulation utilities

### Placeholder Modules (Phases 2 & 3)

**Phase 2 (20%):**
- [TODO] `src/phase2_algorithm/checkpointing_algo.py` - μ-TWO algorithm

**Phase 3 (45%):**
- [TODO] `src/phase3_rewriter/subgraph_extractor.py` - Subgraph extraction
- [TODO] `src/phase3_rewriter/graph_rewriter.py` - Graph rewriting

## How to Use

### 1. Install Dependencies
```bash
source systemProject/bin/activate
pip install -r requirements.txt
```

### 2. Test Basic Training

**ResNet-152:**
```bash
python experiments/train_resnet.py --batch-size 32
```

**BERT:**
```bash
python experiments/train_bert.py --batch-size 16 --seq-length 128
```

### 3. Profile Models (Phase 1)

**Profile ResNet-152:**
```bash
python experiments/train_resnet.py --profile --batch-size 32
```

**Profile with multiple batch sizes:**
```bash
python experiments/train_resnet.py --profile --multiple-batch-sizes
```

**Profile BERT:**
```bash
python experiments/train_bert.py --profile --batch-size 16
```

**With GPU (if available):**
```bash
python experiments/train_resnet.py --profile --device cuda --batch-size 64
```

### 4. View Results

Generated visualizations will be saved to:
- `results/graphs/resnet152/` - ResNet-152 profiling graphs
- `results/graphs/bert/` - BERT profiling graphs

## Next Steps

### Current Status: Phase 1 Development

1. **Test the profiler** on small models first
2. **Verify graph capture** is working correctly
3. **Validate memory tracking** accuracy
4. **Generate baseline statistics** (without activation checkpointing)

### Timeline

- **Weeks 1-4**: Phase 1 - Graph Profiler (code structure ready)
  - Test and debug profiler
  - Collect baseline statistics
  - Generate required deliverables (4a, 4b without AC)

- **Weeks 5-6**: Phase 2 - Activation Checkpointing Algorithm
  - Implement μ-TWO algorithm
  - Decide checkpoint strategy

- **Weeks 7-9**: Phase 3 - Graph Rewriter
  - Extract and replicate subgraphs
  - Insert recomputation nodes
  - Generate final comparisons

## Key Features Implemented

### Graph Profiler
- Captures full computation graph (forward + backward + optimizer)
- Profiles time and memory per operation
- Classifies tensors (parameters, gradients, activations)
- Tracks activation lifetimes
- Generates memory timeline visualizations
- Creates peak memory breakdown charts

### Models
- ResNet-152 with 60M parameters
- BERT-base with 110M parameters
- Configurable for different batch sizes
- Support for CPU/CUDA/MPS

### Analysis Tools
- Activation lifetime analysis
- Memory usage breakdown by tensor type
- Peak memory computation
- Visualization of memory patterns
- Batch size scaling experiments

## Notes

- The profiler uses PyTorch hooks to capture operations
- Memory tracking works best on CUDA (more accurate than CPU)
- Start with small batch sizes to avoid OOM errors
- Phase 2 and 3 modules are placeholder stubs for now
