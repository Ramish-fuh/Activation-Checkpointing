# CS265 Systems Project: Activation Checkpointing in μ-TWO

Implementation of activation checkpointing algorithm to reduce GPU memory consumption during neural network training.

## Project Overview

This project implements activation checkpointing to reduce peak memory consumption during neural network training by trading computation for memory. We target ResNet-152 (vision) and BERT (language) models.

## Installation

```bash
# Activate virtual environment (already created)
source systemProject/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `src/phase1_profiler/` - Computational graph profiler (35%)
- `src/phase2_algorithm/` - Activation checkpointing algorithm (20%)
- `src/phase3_rewriter/` - Graph extractor and rewriter (45%)
- `src/models/` - ResNet-152 and BERT model wrappers
- `experiments/` - Training and profiling scripts
- `results/` - Output visualizations and profiling data

## Usage

### Phase 1: Profile Models
```bash
python experiments/train_resnet.py --profile --batch-size 32
python experiments/train_bert.py --profile --batch-size 16
```

## Timeline

- **Weeks 1-4**: Phase 1 - Graph Profiler
- **Weeks 5-6**: Phase 2 - AC Algorithm  
- **Weeks 7-9**: Phase 3 - Graph Rewriter

## Models

- **ResNet-152**: 152-layer vision model
- **BERT**: Transformer-based language model
