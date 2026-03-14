Questions:

1.	How does PyTorch autograd store activations?
2.	When exactly does memory peak?
3.	How does batch size affect memory?
4.	What layers dominate memory?
5.	How does AC affect iteration time?



More Exploration:
	•	Deep dive into Activation Checkpointing theory
	•	Break down how to profile GPU memory properly
	•	Walk through how to modify PyTorch graph
	•	Or analyze what kind of AC strategy would score highest


Pytorch
	1.	Runs forward normally
	2.	Discards intermediate activations
	3.	During backward:
	•	Re-runs forward of that block
	•	Then computes gradients

Important:
	•	Only works for functions without in-place ops
	•	Doubles compute for that region



    Find why

    1. Resnet50 shows ACT: 0 B at peak: That's because at batch_size=4 the parameter+gradient+optimizer state memory (97.5 + 97.5 + 585 = 780 MB) dominates, and the peak step lands where no activations are alive. 

	 2. Why did `_fused_adam` not appear even on a T4 GPU?
		 Being on a CUDA GPU only means fused Adam is available. It does not mean PyTorch will automatically choose it.
		 The traced optimizer path depends on how Adam is configured:
		 - `foreach=True` traces to `_foreach_*` ops
		 - `fused=True` traces to `aten._fused_adam`
		profiler currently expects `aten._fused_adam`, so the failure was caused by optimizer dispatch choice, not by missing GPU support.