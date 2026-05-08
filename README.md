# Experimental Analysis


## 1. Profiling and Static Analysis Summary

The profiler traces one full training step, including forward, backward, and optimizer updates, and classifies tensors into PARAM, ACT, GRAD, OPT, and OTHER. For both models, the accounting invariants hold:

- `timeline_total_equals_sum_of_components == true`
- `forward_peak_total_equals_sum_of_components == true`
- `overall_peak_total_equals_sum_of_components == true`

### BERT Static Analysis
- Total traced nodes: 2,385
- Tensor-role counts: PARAM 106, ACT 127, GRAD 106, OPT 212, OTHER 1,834
- Checkpointable activations: 127
- Forward candidates: 192
- Forward peak: 160.6 MB without AC, 79.0 MB with AC at batch size 4
- Overall peak: 529.0 MB without AC, 529.0 MB with AC at batch size 4

### ResNet152 Static Analysis
- Total traced nodes: 8,682
- Tensor-role counts: PARAM 467, ACT 776, GRAD 467, OPT 934, OTHER 6,038
- Checkpointable activations: 776
- Forward candidates: 1,295
- Forward peak: 902.9 MB without AC, 853.9 MB with AC at batch size 4
- Overall peak: 2,218.7 MB without AC, 2,218.7 MB with AC at batch size 4

## 2. Peak Memory vs Mini-batch Size

### BERT

| Batch Size | Without AC (MB) | With AC (MB) | Savings (MB) |
|---|---:|---:|---:|
| 4  | 529.0  | 529.0  | 0.0 |
| 8  | 608.4  | 584.4  | 24.0 |
| 16 | 1070.6 | 1002.6 | 68.0 |

![BERT peak memory comparison](/Analyze/Bert/deliverable_peak_memory_vs_batch_size_Bert.png)

**Interpretation:**
- Peak memory increases with batch size as expected.
- AC reduces peak memory more clearly at larger batch sizes.
- For batch size 4, overall peak memory is unchanged, which indicates that non-activation components dominate the total peak at that scale.

### ResNet152

| Batch Size | Without AC (MB) | With AC (MB) | Savings (MB) |
|---|---:|---:|---:|
| 4  | 2218.7 | 2218.7 | 0.0 |
| 8  | 2218.7 | 2218.7 | 0.0 |
| 16 | 3397.3 | 3201.3 | 196.0 |

![ResNet152 peak memory comparison](/Analyze/Resnet/deliverable_peak_memory_vs_batch_size_Resnet152.png)

**Interpretation:**
- ResNet152 shows a stronger memory footprint than BERT overall.
- AC does not change overall peak memory at batch sizes 4 and 8, but it does reduce the peak at batch size 16.
- This suggests that the highest-memory region is partly driven by components outside the forward activations that AC can influence only indirectly.

## 3. Iteration Latency vs Mini-batch Size

### BERT

| Batch Size | Without AC (ms) | With AC (ms) | Overhead (ms) | Overhead (%) |
|---|---:|---:|---:|---:|
| 4  | 30.17 | 34.28 | 4.11 | 13.61% |
| 8  | 44.08 | 50.52 | 6.44 | 14.61% |
| 16 | 80.36 | 81.11 | 0.75 | 0.93% |

![BERT iteration latency comparison](/Analyze/Bert/deliverable_iteration_latency_vs_batch_size_Bert.png)

### ResNet152

| Batch Size | Without AC (ms) | With AC (ms) | Overhead (ms) | Overhead (%) |
|---|---:|---:|---:|---:|
| 4  | 139.33 | 139.34 | 0.01 | 0.01% |
| 8  | 233.34 | 236.22 | 2.88 | 1.24% |
| 16 | 386.07 | 389.66 | 3.59 | 0.93% |

![ResNet152 iteration latency comparison](/Analyze/Resnet/deliverable_iteration_latency_vs_batch_size_Resnet152.png)

**Interpretation:**
- Latency rises with batch size for both models, which is expected.
- AC adds a small runtime cost, but the overhead is modest in all cases.
- For BERT, the overhead is noticeable at smaller batch sizes and becomes negligible by batch size 16.
- For ResNet152, the latency overhead stays under 1.5% for all tested batch sizes.

## 4. Static Profiling Interpretation

The static diagnostics explain the plots:

- The large OTHER category is expected because it includes temporaries, placeholders, and non-activation intermediates that are still alive at the forward peak.
- The ACT category is the main target of checkpointing, and the forward-peak plots show that AC reduces this component as intended.
- The overall peak is not always reduced as much as the forward peak because later-stage components such as optimizer state and backward intermediates can dominate total memory.
- This is why the report should distinguish between **forward peak memory** and **overall peak memory**.

## 5. Conclusion

The experimental results support the checkpointing pipeline:

1. The profiler produces consistent accounting invariants.
2. Tensor classification and liveness analysis are internally coherent.
3. Activation checkpointing reduces forward peak memory, especially as batch size grows.
4. Runtime overhead is small relative to the memory savings.
5. The results are stable across both BERT and ResNet152, so they are suitable for the final report.

