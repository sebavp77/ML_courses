Time series deal with data over time. It can be classify into two different categories:
* **Classification** "Which of these points is an anomaly". The output is discrete
* **Forecasting** Predicts the future value of something. For example the value in the market of some company
![[time_series_structure.png]]

### #uncertainty
There are two types of uncertainties in #time_series prediction: #coconut and #subway

* #aleatoric_uncertainty this type of uncertainty cannot be reduced, it is also referred to as "data" or #subway uncertainty. Example: Let's say your train is scheduled to arrive at 10:08am but very rarely does it arrive at _exactly_ 10:08 am. You know it's usually a minute or two either side and perhaps up to 10-minutes late if traffic is bad. Even with all the data you could imagine, this level of uncertainty is still going to be present (much of it being noise)

* #Epistemic_uncertainty this type of uncertainty can be reduced, it is also referred to as "model" or #coconut  uncertainty, it is very hard to calculate. Example    The analogy for coconut uncertainty involves whether or not you'd get hit on the head by a coconut when going to a beach.
    -   If you were at a beach with coconuts trees, as you could imagine, this would be very hard to calculate. How often does a coconut fall of a tree? Where are you standing?
    -   But you could reduce this uncertainty to zero by going to a beach without coconuts (collect more data about your situation).
-   Model uncertainty can be reduced by collecting more data samples/building a model to capture different parameters about the data you're modelling.

# useful modules or packages

### performance optimized TensorFlow Datasets by
1.  Turning `X_all` and `y_all` into tensor Datasets using [`tf.data.Dataset.from_tensor_slices()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices)
2.  Combining the features and labels into a Dataset tuple using [`tf.data.Dataset.zip()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#zip)
3.  Batch and prefetch the data using [`tf.data.Dataset.batch()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch) and [`tf.data.Dataset.prefetch()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) respectively