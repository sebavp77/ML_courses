These are notes and observations about #kaggle competitions. How to improve, recurrent patterns and other comments.

The following observations come from an article published in **Towardsdatascience.com** 
1. <mark class='red'>Take your time to explore the dataset</mark>: "Understanding the data will save you a lot of time down the road and will help you in making informed decisions as to which model to try and which models to avoid." Some of the most useful and starting point features are:
	1. check the **distribution of features** Try to see if the features are normally distributed or if there are skewed distributions.1.  Check the data types of features
	2.  Check the mean, median, and std of features
	3.  Check the number of features, size of the dataset
	4.  Check if there are any nullable/empty fields
	5.  Check the correlation of features, highly correlated features can sometimes be dropped
2. <mark class='yellow'>Read papers about the field of the competition</mark> : For example, if you are doing a competition about detecting skin cancer in images, try to find the most recent papers doing the most similar task (which is probably going to be computer vision in bioinformatics for example).
3. <mark class='yellow'>Start experimenting with models</mark> : keep track of your models, the hyperparameters you used and the performance of each one of them. "A very common mistake here is to not track your experiments and get lost into a chaos of notebooks. You have to be systematic about your experiments, note down the model you are going to use and the range of hyperparameters you are going to be testing with. It is very important to establish a solid baseline that you can build on top of." <mark class='red'>Don't start with a Neural Network</mark>, start with simple models, like Bayes or regression and increment the complexity of your models.
<mark class='red'>Also, you have to learn how to use </mark> **gradient boosting machines** <mark class='red'>because they are absolute beasts when it comes to performance in these Kaggle competitions. They are also very easy to use and several ones as Light GBM give you quite meaningful results such as the importance of features</mark>  
4. Ensemble, ensemble, ensemble ( #ensemble)
	Ensembling is honestly the secret key to getting a good score in Kaggle competitions. I used to always forget about doing this. After you run your experiments, don’t simply submit the predictions for your best model. Submit an ensemble of your best 3 or maybe even 5 models. And always train your models using 3 or 5 K-folds.