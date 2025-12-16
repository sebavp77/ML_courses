
# Links
* [[Deep learning]]
* [[Kaggle competitions]]
* [[Combining models]]
* [[Resources]]
* 


# General aspects

The following is a basic workflow to guide how to track problems with machine learning

![[steps_machine_learning.webp]]

# 1. Problem definition
Mainly, there are 2 types of problems in machine learning: #supervised and #unsupervised problems. Each one of this has subdivisions. For example, #supervised problems can be divided into #classification and #regression. In the next image is depite the subdivision 

![[machine_learning_type_problems.png]]

# 2. Data
This is probably one of the most important steps in any data science problem, the data. It is also, one of the steps that I am probably going to expend more time. It is super important to have a clean dataset, identify what I have, visualize, and organize it in a fashion way that it is split it into train and test data sets

![[Typical-Data-Analytics-Workflow.gif]]

first I have to read in the data. Then I can look for nans, wrong entries, etc. If it is possible, reduce the dimensionality or consider the most important varaibles. The final step is obtain statistics about my data sets as for example length, shape, number of samples for each label ( #classification ). I would say, that parallel to each step goes visualization. It is crucial to visualize as much as possible to have a better idea about what we are doing.
the data can be #split it in: ## Split data into:
		* **Training set**
		* **Validation set**
		* **Test set**

### Batch and prepare datasets: 

When we have our data, it is usual that the size is to big to fit in memory, so we explit the data into #batchs, this is a common practice. There are multiple of ways to do this. Nevertheless, it is important to keep in mind that speed is an important metric.
One approach is by using #tensorflow, this exaple is related with image data, but it works for any kind of tensor. We are going to use the  [`tf.data` API](https://www.tensorflow.org/api_docs/python/tf/data) . Specifically, we are going to be using:
-   [`map()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) - maps a predefined function to a target dataset (e.g. `preprocess_img()` to our image tensors)
-   [`shuffle()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) - randomly shuffles the elements of a target dataset up `buffer_size` (ideally, the `buffer_size` is equal to the size of the dataset, however, this may have implications on memory)
-   [`batch()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch) - turns elements of a target dataset into batches (size defined by parameter `batch_size`)
-   [`prefetch()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) - prepares subsequent batches of data whilst other batches of data are being computed on (improves data loading speed but costs memory)
-   Extra: [`cache()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache) - caches (saves them for later) elements in a target dataset, saving loading time (will only work if your dataset is small enough to fit in memory, standard Colab instances only have 12GB of memory)
```info
Original dataset (e.g. train_data) -> map() -> shuffle() -> batch() -> prefetch() -> PrefetchDataset
```
An example of this function is:
```jupyter
# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

```
In this case **preprocess_img** is a function made by the user to perform something on the data

* ##### Another approach, again using [`tf.data` API](https://www.tensorflow.org/api_docs/python/tf/data) is as follows:
In this ocassion our input data is not in format of tensor, so we need to convert it.
```jupyter
# Turn our data into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
```
Now that our data has the right format we can use the same functionalities as before:
```jupyter
# Take the TensorSliceDataset's and turn them into prefetched batches
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```
Notice that in this ocassion we didn't shuffle the input data

# 3. Evaluation
If we want to assess our model, we need to define a metric to use. So according to the type of problem we have, what is the best metric to use?
In order to anser this you can take a look to the following image
![[metrics_machinelearning.png]]
In this nice image you can see that if you have already idetified your problem, the election of the metric evaluation is a easy step. But, there are many options inside each type of problem, so what should we choose?
The answer is that it depends on the goal of our model, each metric is used to assess a different goal.

# 5. Modelling
We can split #modellin intro three main steps:

1. Choosing and training a model: You will use the #training data.
	Some tips whe choosing a model: if you hae #structured data #random_forest and #decision_trees work bette, but I have unstructured data (like images or voice) #deep_learning and #transfer_learning perform better
1. Tuning a model: you will use the #validation data
2. Model comparison: you will use the #test data

# Preprocessing data
A common practice when working with neural networks and ML algorithms is to make sure all of the data you pass to them is in the range 0 to 1.

This practice is called **normalization** (scaling all values from their original range to, e.g. between 0 and 100,000 to be between 0 and 1).

There is another process call **standardization** which converts all of your data to unit variance and 0 mean.

These two practices are often part of a preprocessing pipeline (a series of functions to prepare your data for use with neural networks), and they recieve the name of #feature_scaling

In practical terms you can do this with <mark class='yellow'>Scikit-Learn</mark>, but there are different functions to choose from inside Scikit-Learn: _MinMaxScaler_, _RobustScaler_, _StandardScaler_, and _Normalizer_. So, what is the right one for me?, it depends on my **model type** and **feature values**.

Let's consider the following input data with different distributions

![[distributions.png]]
Let's consider each one of the different options

### MinMaxScaler
For each value in a feature, [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler) subtracts the minimum value in the feature and then divides by the range. The range is the difference between the original maximum and original minimum.
It preserves the **shape** of the original distribution. It doesn't affect **outliers** 
The output #scale is between 1 and 0
![[minmax_scaler.png]]

### RobustScaler
[RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value). As before, it doesn't affect the shape but a difference is that it doesn't have a predetermined output scale **(the output is not between 1 and 0)**
![[robust_scaler.png]]
### StandardScaler
<mark class='yellow'>StandardScaler is the industry’s go-to algorithm.</mark>
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) standardizes a feature by subtracting the mean and then scaling to unit variance. Unit variance means dividing all the values by the standard deviation.
It meaks the #mean distribution approximately 0 and unit #variance
![[standard_scaler.png]]
### Normalizer
[Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) <mark class='yellow'> works on the rows, not the columns</mark> 
By default, L2 normalization is applied to each observation so the that the values in a row have a unit norm. _Unit norm_ with L2 means that if each element were squared and summed, the total would equal 1.

**This type of normalization is not really useful**

### In conclusion
* #StandardScaler if you want each feature to have zero-mean, unit standard-deviation
* #MinMaxScaler if you want to have a light touch. It’s non-distorting
* #RobustScaler if you have outliers and want to reduce their influence.


## Which models perform better with scaled data
* linear and logistic regression
-   nearest neighbors
-   neural networks
-   support vector machines with radial bias kernel functions
-   principal components analysis
-   linear discriminant analysis


## Sklearn.base.BaseEstimator

This is the base class for all estimator in scikit-learn. 

## Python Decorators
In order to understand decorators and how they work, we are going to start seeing some examples about what they do and later write in words what we have learned from those examples.

Let's start by considering a function that takes as argument another function and the calls it n times (let's fix n to 2)

```jupyter
def repeat(fn):

    fn()

    fn()

def hello_world():

    print("Hello world!")

repeat(hello_world)

```

Continuing with the idea, we can make a function to execute another function (what we did above) or to return another function
```jupyter
def repeat_decorator(fn):

    def decorated_fn():

        fn()

        fn()

    # returns a function

    return decorated_fn

def hello_world():

    print ("Hello world!")

hello_world_twice = repeat_decorator(hello_world)

# call the function

hello_world_twice()
```
In this case inside the function `repteat_decorator(fn)` which takes as input a function another function is defined `decorated_fn()`, but in this case it doesn't take any argument but uses the one was used in the parent function. We can extend this idea not to only one function but in the case, for example we want to write different function within one, for example to call 1, 2, and 5 times another function.
**Note** in case more functions are defined in `repeat_decorator` it will call only the first function

In the example above we named the new variables as *hello_world_twice* but we can give any name to this variable even the same name of an already defined function as for example *hello_world*

```jupyter
  
def repeat_decorator(fn):

    def decorated_fn():

        fn()

        fn()

    # returns a function

    return decorated_fn

def hello_world():

    print ("Hello world!")

hello_world = repeat_decorator(hello_world)

# call the function

hello_world()
```
In this case the variable name *hello_world* is overwrite and it goes from being a function that prints "hello world!" to a function named "decorated_fn" which calls twice the function that was used as argument before, in this case `hello_world`. 

OK, so this can look confusing, at least for me, but what is happening is that we change the functionality of a function. At the beginning this function would write "hello world!" but now it is writing two times "hello world!".

```jupyter
def repeat_decorator(fn):

    def decorated_fn():

        a = fn()

        b = 'This is a new function of the original ' + a

        print(b)

    # returns a function

    return decorated_fn

def hello_world():

    return ("Hello world!")

hello_world = repeat_decorator(hello_world)

# call the function

hello_world()
```
In this new example if you call `hello_world()` you will obtain `This is a new function of the original Hello world!` and not `Hello world!` because the original functionality of the function has been changed.

==**Now let's make use of decorators**==
```jupyter
# function decorator that calls the function twice

def repeat_decorator(fn):

    def decorated_fn():

        fn()

        fn()

    # returns a function

    return decorated_fn

# using the decorator on hello_world function

@repeat_decorator
def hello_world():

    print ("Hello world!")

# call the function

hello_world()
```
*Explanation of the syntax*  the part `@repeat_decorator` before the function definition means that the function that is defined below will be passed to the function `repeat_decorator` and reassign its name to the output. The exact same behavior as before. 

**CONCLUSION** You define a function, and you use a decorator when you want to extend or alter the functionality of the function you have previously defined taking advantages or other functions that accept as input a function.

