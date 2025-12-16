#transfer_learning consist in using an already existing NN in your problem. There are two different approaches: you can download the NN and used it as it is ( #feature_extraction) or you can download the NN and retrain certain layers to adapt it better to your problem ( #fine_tuning)

* To find different NN available to be used you can go to [tfhub.dev](https://tfhub.dev/). 

#NOTE 🗝️: <mark class='yellow'>You can see a list of state of the art models on </mark>[paperswithcode.com](https://www.paperswithcode.com/)
( #state_of_the_art_models)

To download the models you can use the module: 
```jupyter
import tensorflow_hub as hub

# Original: EfficientNetB0 feature vector (version 1)
model_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False, # freeze the underlying patterns
                                           name='feature_extraction_layer',
                                           input_shape=IMAGE_SHAPE+(3,)) # define the input image shape
```
and then you can create the whole model either with **Sequential** or with **Functional**

* There is also another option to use #pretrained models and it is using:
```jupyter
# 1. Create base model with tf.keras.applications
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
```
In this case we are using the model **EfficientNetB0** with the top layer not included because we have a different output than in the actual model

* Other interesting #functions or #layers are, for example:
	* #Pooling layers as  [`tf.keras.layers.GlobalAveragePooling2D()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D) or [`tf.keras.layers.GlobalMaxPooling2D()`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool2D?hl=en) They reduce the dimension of the input tensor.
	They turn a 4D tensor into a 2D tensor by averaging the inner axes

# Passing image batch to our model

Another function we can use to pass our image batch to our model in an efficient way is by using 
['image_dataset_from_directory'](https://stackoverflow.com/questions/71704268/using-tf-keras-utils-image-dataset-from-directory-with-label-list) , you can see an example below
```jupyter
IMG_SIZE = (224, 224)
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_10_percent,
                                                                            label_mode="categorical",
                                                                            image_size=IMG_SIZE)
```
