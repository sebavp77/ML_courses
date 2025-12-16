The normal workflow when dealing with input data in the format of text is:

```info
Text -> turn into numbers -> build a model -> train the model to find patterns -> use patterns (make predictions)
```

where the part **turn into numbers** involve the new and additional step in comparison with numerical data.

<mark class='yellow'>Remember</mark>  📣: Machine Learning algorithms only accept numerical data

The proccess of turning text into numbers receive the name of #tokenization and #vectorization or #embedding

**Tokenization** is when you conver each word or character into a number
**Vectorization** is when, once you have these numbers, you expand the dimension of each number into a vector which reflects the relationship between the words or characters and their surrounding characters or words.

When dealing with this, there are two main approaches:
1. You can create your own token and vectorization by using sckit functions
2. You can use transfer learning and use an already existing net

## Tokenization ( #tokenization )

One way of doing this is by using the preprocessing layer [`tf.keras.layers.experimental.preprocessing.TextVectorization`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization). This takes the following parameters
-   `max_tokens` - The maximum number of words in your vocabulary (e.g. 20000 or the number of unique words in your text), includes a value for OOV (out of vocabulary) tokens.
-   `standardize` - Method for standardizing text. Default is `"lower_and_strip_punctuation"` which lowers text and removes all punctuation marks.
-   `split` - How to split text, default is `"whitespace"` which splits on spaces.
-   `ngrams` - How many words to contain per token split, for example, `ngrams=2` splits tokens into continuous sequences of 2.
-   `output_mode` - How to output tokens, can be `"int"` (integer mapping), `"binary"` (one-hot encoding), `"count"` or `"tf-idf"`. See documentation for more.
-   `output_sequence_length` - Length of tokenized sequence to output. For example, if `output_sequence_length=150`, all tokenized sequences will be 150 tokens long.
-   `pad_to_max_tokens` - Defaults to `False`, if `True`, the output feature axis will be padded to `max_tokens` even if the number of unique tokens in the vocabulary is less than `max_tokens`. Only valid in certain modes, see docs for more

## Embeddings ( #embedding )

As we can use tensorflow to create our tokenz we can use it to embed our tokenz. In this case the preprocessing layer is [`tf.keras.layers.Embedding`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) , an interesting thing to point out is that this layer is trained and the vectors will change as the whole NN is trained. The main parameters are:
-   `input_dim` - The size of the vocabulary (e.g. `len(text_vectorizer.get_vocabulary()`).
-   `output_dim` - The size of the output embedding vector, for example, a value of `100` outputs a feature vector of size 100 for each word.
-   `embeddings_initializer` - How to initialize the embeddings matrix, default is `"uniform"` which randomly initalizes embedding matrix with uniform distribution. This can be changed for using pre-learned embeddings.
-   `input_length` - Length of sequences being passed to embedding layer.
An example use of this layer is:
```jupyter
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=128, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             input_length=max_length, # how long is each input
                             name="embedding_1") 

embedding
```


<mark class='yellow'> Observation </mark> 📣: When dealing with multi class classification you can use **one hot encodding** ( #onehotencodding) or **label encoded** ( #labelencoded). It is important to note that TensorFlow's CategoricalCrossentropy loss function likes to have one hot encoded labels

