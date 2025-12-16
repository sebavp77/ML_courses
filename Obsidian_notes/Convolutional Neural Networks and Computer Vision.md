The structure of #CCN (Convolutional Neural Network) is the same as any other neural network

**Input** --> **hidden layers** --> **output** 

the difference is the type of layers, in this case they are convolutional and the use and function is different. 
In the following image you can see the basic structure of a #CNN
![[basicstructure_CNN.png]]
_Remember:_ In this type of problems, for example, dealing with #images, the #feature_scaling is important.

**Observation** when passing data from the folder to the model, there are multiple ways to improve the speed at which this happens. A normal approach will be specify the folder for the training data and give this to the NN, but there are built-in functions that optimize this process.
some examples of these are:

* #image_generator: from tensorflow.keras.preprocessing.image import ImageDataGenerator
```jupyter
# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time 
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               seed=42)
            
.
.
.
# Fit the model
history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
```

**Observation 🗝️** Each layer requires a different input, so take this into account

**Data augmentation** ( #data_augmentation) is the process of altering our training data, leading to it having more diversity and in turn allowing our models to learn more generalizable patterns. Altering might mean adjusting the rotation of an image, flipping it, cropping it or something similar.

A really nice workflow of the general procces is shown in the next Image
![[workflow_image_classification.png]]
 