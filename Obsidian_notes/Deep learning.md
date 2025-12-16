* [[Time series]]
*  [[Convolutional Neural Networks and Computer Vision]]
* [[Transfer learning]]
* [[Natural Language Processing]]
* [[Recurrent Neural Networks]]
* [[Dynamic Graph CNN (Edge Conv)]]
* [[Graph Neural Networks (GNN)]]
* 

# Steps in modelling with TensorFlow:

1. Creating a model: piece together the layers of a neural network yourself (using the [Functional](https://www.tensorflow.org/guide/keras/functional) or [Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)) or import a previously built model (known as transfer learning).
2. **Compiling a model** - defining how a models performance should be measured (loss/metrics) as well as defining how it should improve (optimizer).
3. 1.  **Fitting a model** - letting the model try to find patterns in the data (how does `X` get to `y`).

	If you want to #plot the model you have created you can use 
```ad-info
	from tensorflow.keras.utils import plot_model
	plot_model(model, show_shapes=True)
```

# Saving the model
You can save a TensorFlow/Keras model using [`model.save()`](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model).
There are two ways to save a model in TensorFlow:

1.  The [SavedModel format](https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format) (default).
2.  The [HDF5 format](https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format).


# Classification problems

There are two types of #classification problems: #binary and #multiclass classification. When building a NN for this kind of problem the steps are almost the same for binary and multiclass classification. Nevertheless, one difference is the election of the #output_activation layer:
	* Binary: Sigmoid
	* Multiclass: Softmax
Another difference is the #loss_function:
	* Binary: [`tf.keras.losses.BinaryCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) in TensorFlow
	* Multiclass: [`tf.keras.losses.CategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) in TensorFlow

### Evaluation methods: ( #evaluation_methods)

![[evaluation_methods_classification.png]]

* There is a function of Scikit-Learn than enable us to make a #classification_report of our results
```jupyter
from sklearn.metrics import classification_report
print(classification_report(y_labels, pred_classes))
```

# Hyperparameters
Hyperparameters are parameters of your model that you can tune (change) to improve the results. Some of them are:
* Learning rate ( #learning_rate)
* Number of epochs ( #epochs)
* Number of layers
* Activation for layers 

# Callbacks
Callbacks are (really) useful functions that can be implemented when you are fitting (training) your model. Some of them are:
* **Finding the best learning rate** ( #learning_rate ):  This callback enables you to schedule how the learning rate will change
tf.keras.callbacks.LearningRateScheduler(    schedule, verbose=0)
```jupyter
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20)) # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch
```
* **Creating a checkpoint** ( #ModelCheckpoint): The [`ModelCheckpoint`](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) callback gives you the ability to save your model, as a whole in the [`SavedModel`](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) format or the [weights (patterns) only](https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights) to a specified directory as it trains. <mark class='yellow'> This is helpful if you think your model is going to be training for a long time and you want to make backups of it as it trains </mark>
```jupyter
# Setup checkpoint path
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt" # note: remember saving directly to Colab is temporary

# Create a ModelCheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, # set to False to save the entire model
                                                         save_best_only=False, # set to True to save only the best model instead of a model every epoch 
                                                         save_freq="epoch", # save every epoch
                                                         verbose=1)
```

* **Early stopping callback** ( #earlystopping): it monitors a specified model performance metric (e.g. `val_loss`) and when it stops improving for a specified number of epochs, automatically stops training ([`EarlyStopping` callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).)
```jupyter
# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=3) # if val loss decreases for 3 epochs in a row, stop training
```

* ***Learning rate reduce on plateau** ( #ReduceLROnPlateau): <mark class='red'>the learning rate is the most important model hyperparameter you can tune</mark> 
[ReduceLROnPlateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau)  callback montiors a specified metric and when that metric stops improving, it reduces the learning rate by a specified factor
An example use is
```jupyter
# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)
```


### How to include the Callbacks in the training
Fit the model (passing the lr_scheduler callback)

history = model_9.fit(X_train,  y_train, epochs=100, callbacks=[lr_scheduler])