## Attention ( #attention)

**Definition** It describes a *weighted average* of (sequence) elements with the weights dynamically computed *based* on an *input query* and *element's keys*. In other words, we want to decide ***dynamically*** on which elements put more attention than others. An attention mechanism has **4** elements we need to define:

1. ***Query:***  Feature vector that describes what we are looking for in the sequence
2. ***Key:*** For each input element, we have a key which is again a feature vector. This feature vector describes roughly what the element is offering or when it might be important. The keys should be designed such that we can identify the elements we want to pay attention to based on the query.
3. ***Values:*** For each input element, we also have a value vector. This feature vector is the one we want to average over
4. ***Score function:*** It rates which elements we want to pay attention to. It takes the query and a key as input values and output the score attention weight of the query-key pair. 


## Scaled Dot Product Attention

This is the *core concept* behind *self-attention*. 
**Definition:** ***Self-attention***: each sequence element provides a key, value and query.

Having a set of queries ***Q***, keys ***K*** where both $\epsilon \; R^{T \times d_k}$, and values ***V*** ( $\epsilon \; R^{T \times d_v}$). Where $T$ is the sequence length, and $d_k$ and $d_v$ are the dimensionalities of queries/keys and values, respectively.
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

The matrix multiplication $QK^T$ performs the dot product for every possible pair of queries and keys, resulting in a matrix of the shape $T \times T$ 

### Multi-Head Attention

Often there are multiple different aspects a sequence element wants to attend to and a single weighted average is not a good option for it. This is why we extend the attention mechanism to multiple heads, i.e. *multiple different query-key-value triplets* on the same features

**How it works:** Given a *query, key and value matrix* we transform those into ***h sub-queries, sub-keys, and sub-values***, which we pass through the scaled dot product independently. Afterwards, we concatenate the heads and combine them with a final weight matrix 