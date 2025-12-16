This topic is related with data augmentation, in which we made slightly modifications to the original dataset as for example: crop, rotation, zoom, flip, with the goal to add more variety to our dataset and prevent overfitting. 

Now, instead of focusing on the way we **train** our model, test-time augmentation deals with the way we **test**

It performs random modifications to the test images. Instead of showing the clean images only once to the trained model, we will show it the augmented (modified ) images several times. Later the predictions for the same original image are averaged and that is the final guess.