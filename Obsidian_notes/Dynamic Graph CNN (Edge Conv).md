
# References:
* https://medium.com/@sanketgujar95/dynamic-graph-cnn-edge-conv-2582c3eb18d8#:~:text=What%20is%20EdgeConv%3F,edges%20from%20each%20connecting%20vertex.
* [ Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)

# Edge Conv

( #edge_conv) Its appealing property is that it incoporates *local neighborhood information* and can be stacked or recurrently applied to learn *global shape properties*.
* It generates **edge features** which describe the relationship between a point and its neighbors 

*==Property==* Desgined to better capture local geometric functions and be invariant to the ordering of neighbors, and thus, permutation invariant.
	**Interpretation** It gives information about local relationships, but it is not influence on how the neighbors are located or order.
