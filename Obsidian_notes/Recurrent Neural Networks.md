
We are looking for a model that has the following features:
1. Handle #variable_length sequences (inputs of different lenght, like sentences with different amount of words)
2. Track #long_term dependencies (how one word depends on the context, previous words)
3. Maintain information about #order
4. #share_parameters across the sequence

![[basic_RNN.png]]

Recurrent Neural Networks ( #RNN) are particularly good for #sequence of data (as can be words). For example, in the figure below, you can see how the neuron (green box) is feeded with an input (blue circle) and the output of the previous neuron. This forms a sequence 

![[example_RNN.png]]

The first image illustrates how RNN keep track of past time steps and then, this information is pass to the next time step. So let's take a look about how this works. In the following image is depicted the working principle. The RNN keep information about the previous state in **h_t**. In the next time step the RNN cell will recieve the regular input ( **x_t**) as a forward neural network, but with the addition of this new varibale **h_t**. But, How does this variable update?
the update of **h_t** is obtained with the equation shown in the right part of the image. It depends on a fuction parameterized by w, the old state and the the new input vector.

**NOTE** the same function and parameters ( weights and bias) are used at every time step.

![[workingprinciple_RNN.png]]

![[activation_RNN.png]]

Ok, pretty well, so now we know the basic idea and concepts about RNN, but how do we train this kind of neural networks? Let's find the answer to that question

## Back Propagation Through Time ( #BPTT )
It is a similar idea that in **Forward neural netwotks** but we are going to do two things:
1. Forward propagate trhough the network
2. 2.1. Backward propagate at each indivual time step
	 2.2 Backward propagate for all time steps

![[forward_backward_propagation_RNN.png]]

# Long Short Term Memory ( #LSTM)

![[LSTM_structure.png]]

In the previous image you can see the structure of a LSTM. In principle it works as a RNN but, as you can see, it is slightly more complex. It contains different interacting layers. So each LSTM cell contains within it a set of different layers. 
The key feature of this cell is the #gate . A gate let information through. So in this case, the #sigmoid function is converting the input to a value in between 0 and 1.

![[gates_LSTM.png]]

In the following image it is depicted the <mark class='yellow'>most important principles</mark> of a #LSTM. It performs **4 steps**

1. **Forget**: Forget irrelevant information about the current ( **x_t** ) and past state (  **h_t-1**)
2. **Store**: the most important information
3. **Update**: according to the last two previous steps, it stores the most relevant information
4. **Output**: Finally, it gives the output ( **h_t**) 
**Note** C_t works as a creterion to select whether information is relevant or not, it is also used to compute the gradient (back propagating)
![[LSTM_cell_structure.png]]

### Key concepts
1. Maintain a **separate cell state** from what is outputted
2. Use **gates** to control the flow of information
3. Backpropagation through time with **uninterrupted gradient flow** (c_t)

# Gated Recurrent Unit ( #GRU)
GRU (Gated Recurrent Unit) aims to solve the **vanishing gradient problem** which comes with a standard recurrent neural network.

#### what makes them so special and effective?

To solve the vanishing gradient problem of a standard RNN, GRU uses, so-called, **update gate and reset gate**. These are two vectors which decide what information should be passed to the output. The **special thing** about them is that they can be trained to keep information from long ago, without washing it through time.

Let's take a look to one single cell
![[GRU_cell.png]]

$z_t$ is the updated state from the present input ( $x_t$) and the previous step ( $h_{t-1}$), this is added and pass it through a sigmoid activation, this activation selects how much of the previous and pass information is passed.

**reset gate**: this is the $r_t$ variable and it is computed exactly as the above step, it is the sum of $x_t$ and $h_{t-1}$ and then it goes through a sigmoid activation function. The difference with the previous step is on the **weights**

**Current memory content**: This is the $h_t^{'}$ and it is the result from:
	1. $r_t$ and $h_{t-1}$ are multiply element wise. This selects how much of the previous information ($h_{t-1}$) is keep it (remember that $r_t$ is a vector between cero and 1)
	2. Then the result of this product is added to the input (current vector state $x_t$)
	3. Finally, the result is the argument of a tanh to add nonlinearity 
$$
h_t^{'} = tanh(Wx_t + r_t \cdot Uh_{t-1}) 
$$
Where W and U are weights.

**Update gate** (Final memory at current time step):  The network needs to calculate the actual state ($h_t$). In order to do that it determines what to collect from the current memory state ($h_t{'}$) and what from the previous steps ($h_{t-1}$)
	1. It multplies what is keep it from the past with the current state ($z_t$) with the nonlinearity of what they want to forget ($h_t^{'}$)
	2. By other hand, the past state ($h_{t-1}$) is element wise multplied by the inverse of what you want to remember (1-$z_t$) 
	3. Finally, the previous two steps are added and we obtain the actual state
$$
h_t = z_t \cdot h_t^{'} + (1-z_t) \cdot h_{t-1}
$$
_observation_: In this ocassion there are no weights, the network doesn't control how much of this goes to the end.
