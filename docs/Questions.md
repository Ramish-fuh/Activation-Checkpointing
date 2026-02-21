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