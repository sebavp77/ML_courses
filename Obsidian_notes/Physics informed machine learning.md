## Reference
@article{karniadakis2021physics,
  title={Physics-informed machine learning},
  author={Karniadakis, George Em and Kevrekidis, Ioannis G and Lu, Lu and Perdikaris, Paris and Wang, Sifan and Yang, Liu},
  journal={Nature Reviews Physics},
  volume={3},
  number={6},
  pages={422--440},
  year={2021},
  publisher={Nature Publishing Group UK London}
}


## How to embed physics in ML
There are three different pathways that can be followed separately or in tandem to embed physics in ML models: *observational biases*, **inductive biases**, ==learning biases==

### Observational biases
Given enough data to cover the input domain for learning, machine learning models have demonstrated incredible power in achieving accurate interpolation between the dots.
The idea is that data obtained from observations already have the underlying physical principles.
For this type of biases large amount of data is necessary to reinforce the model such that respect physical principles as symmetries and conservation laws.


### Inductive biases
The idea of this bias is to design special architectures that respect symmetries and conservation laws. An example of this is NN which respect invariance along the groups of symmetry and distributed pattern representation found in natural images.
**How is this?:**  If you think about a natural image, let's imagine a dog in a park and I handle this image to you, it doesn't matter in which angle the dog is when you receive the image, for example the dog could be above and the sky below, or the dog and sky parallel to each other. With a pair of rotations you could place the image in a more natural look. Nevertheless, the image is the same regardless of their relative rotation to you. NN are able to capture this symmetries and distributed patterns. You can feed a NN with any rotation of an image and it will be able to recognize the correct image.

Despite of their effectiveness, these approaches are limited to simple and well defined physics or symmetry groups. Additionally, they extension to other tasks is often complicated as the laws and invariances of many physical systems are poorly understood or hard to implicitly encode in a natural architecture.

specialized NN to solve differential equations can be obtained modifying NN architectures to satisfy boundary conditions.  Additionally if some information about the PDE or the system is known it is possible to encode them in network architectures for example even/odd symmetries, energy conservation, high frequencies.

### Learning bias

This is a more soft angle in doting a NN with physics. Instead of designing a NN architecture that capture the physics, the loss function is used in conventional NN penalizing when the appropriate physics is not followed..
This approach can be thought as two constrains: matching the training data and satisfying a set of physical constraints, for example conservation of mass, momentum.

## Connections to kernel methods
Analyzing NN though the lens of kernel methods could have considerable benefits, as kernel methods are often interpretable and have strong theoretical foundations, which in consequence could help us to understand when and why deep learning methods may or may not succeed.

## Merits of physics informed learning
In the following it is discussed in more detail for which scenarios the use of PINNs may be advantageous and highlight these advantages in some prototypical applications

### Incomplete models and imperfect data
examples include when the boundary conditions are not known or where some parameters in the PDE are unknown.
When dealing with imperfect models or data it is beneficial to integrate the Bayesian approach with physics informed learning for uncertainty quantification

### Strong generalization in small data regime
by enforcing or embedding physics, deep learning models are constraint to lower dimensional manifold. Additionally, it is capable of **extrapolation** not only interpolation

## Software
Some of the specifically designed software libraries for physics informed ML are present in the figure
![[table_1_libraries_physics_informed_ML.png]]
__solver__: The user only needs to define the problem and the solver will deal with all underlying details and solve the problem
**Wrapper**: they wrap low level functions (as TensorFlow) into high level functions to facilitate the implementation of physics informed learning and users still need to implement all the steps to solve the problem

* DeppXDE and SimNet use physics as soft penalty (in the loss function) 

## Which model, framework, algorithm to use?
given a physical system and/or governing law and some observational data, which ML framework should I use? The choice intimately depends on the task to be tackled. 
*==PINNS are typically used to infer a deterministic function that is compatible with an underlying physical law when a limited number of observations is available==*
Some list of approaches according to the main characteristics of the problem are:
* Multi layer perceptron architectures: general application, not encode any specialized inductive biases
* Convolutional NN: suitable for gridded 2D domains
* Fourier Feature networks: suitable for PDEs whose solution exhibits high frequencies or periodic boundaries
* Recurrent architectures: suitable for non-Markovian and time discrete problems

the success or failure on applying physics informed to a model or architecture can be resume on:
* the strength of the inductive biases: how well the architecture of the ML model accounts for the physics
* how well the collected data represents the underlying operators or physics laws (loss function)
* How complicated is the function you want to approximated

## Current limitations
NNs have problems learning high frequency functions (think of an image, let's say a dog, the high frequency features of this image will be each hair, the details of the nose, the color inside the eye which is non uniform)

* In contrast to classical ML algorithms where the first order derivative is required for gradient descent, physics informed ML require high order derivatives. Currently, their efficient implementation is not well supported in popular software frameworks as TensorFlow or PyTorch
* The effectiveness of a physic informed ML algorithm to solve PDE can be though or frame into three questions:
	* can a network approximate a solution to PDE with any accuracy
	* can one attain zero or very small training loss
	* does smaller training error mean more accurate predicted solutions?
## Comments
* Combination of DeepONets and PINN can achieve accurate predictions with extrapolation in Multiphysics applications
* study what is a kernel method 

## References to check
* 33: physics based model of bond-order potential with an NN and divide structural parameters into local and global parts to predict interatomic potential energy surface in large-scale atomistic modelling. 
* 13: DeepONets have been demonstrated as a powerful tool to learn nonlinear operators in a supervised data-driven manner.
* 7: proposes a discrete time NN method for solving PDE that is inspired by an implicit Runge-Kutta integrator. it allows very large time steps and lead to solutions of high accuracy
* 145: molecular simulations. NN architecture proposed to represent the potential energy surface for molecular dynamics simulations. The translational, rotational and permutational symmetry of the molecular system is preserved,