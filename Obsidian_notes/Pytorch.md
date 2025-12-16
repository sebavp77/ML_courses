
## Pytorch Workflow
![[pytorch_workflow.png]]
Figure 1. It shows the basic and general steps when dealing with a ML problem

1. Turn the data into tensors
2. Build or pick a petrained model: select a loss function and optimizer
	1. Build a learning loop
3. Fit the model to the data and make predictions
4. Evaluate the model 
5. Improve the model

### Pytorch model building essentials

Pytorch has ***4*** essential modules you can use to create almost any kind of neural network.
These are *torch.nn*, *torch.optim*, *torch.utils.data.Dataset*, *torch.utils.data.DataLoader*
* **torch.nn** It contains all of the building blocks for computational graphs ( a series of computations executed in a particular way)
* **torch.nn.Parameter** Stores ***tensors*** that can be used with ==nn.Module==. If ==**requires_grad=True**== gradients are calculated automatically 
* **torch.nn.Module** ==The base class for all neural network modules==, all the building block for neural networks are subclasses. If you are building a neural network in Pytorch, your models should subclass ==*nn.Module*==. *Requires a ==forward()== method be implemented*
* **torch.optim** It contains various optimizations algorithms 

### Making predictions using `torch.inference_mode()`
When we pass data to our model it will go through the model's ==forward()== method and produces a result using the computation we have defined.

* There are 3 things to remember when making predictions with a trained Pytorch model
	1. Set the model in evaluation mode ==(model.eval())==
	2. Make the predictions using the inference mode context manager (==***with torch.inferece_mode():***==)
	3. All predictions should be made with objects on the same device (e.g. data and model on GPU only or data and model on CPU only)
### Saving and loading a PyTorch model
There are 3 possible methods when saving and loading models
![[save_load_pytorch.png]]
==The [recommended way](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) for saving and loading a model for inference (making predictions) is by saving and loading a model's `state_dict()`.==
Let's see how we can do this
1. Create a directory for saving models to called *models*  using Python's *pathlib* module
2. Create a file path to save the model to 
3. Call *torch.save(obj,f)* where *obj* is the target model's *state_dict()* and *f* is the file name or where to save the model 

## Neural Network Classification

### Binary cross-entropy 
Pytorch has *two binary cross entropy* implementations 
1. *[`torch.nn.BCELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)* Creates a loss function that measures the binary cross entropy between the target (label) and input (features).
2. *[`torch.nn.BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)* This is the same as above except it has a sigmoid layer ([`nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)) built-in

**Which one should you use?**

The [documentation for `torch.nn.BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) states that it's more numerically stable than using `torch.nn.BCELoss()` after a `nn.Sigmoid` layer.

==Note: For advanced usage you should separate the loss from the activation==

## Computer Vision

![[pytorch_compute_vision.png]]

