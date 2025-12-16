There are different approaches for doing cross validation (CV). Nevertheless, all of them share the same basic idea or steps:
1. Divide the dataset into two parts: for training and for testing
2. Train the model on the training set
3. Validate the model on the test set
4. Repeat the steps 1-3 n number of times. 

Among the different approaches for doing CV, we can mention:
1. Hold-out
2. k-folds
3. Leave-one-out
4. Leave-p-out
5. Repeat K-folds
6. Nested K-folds
7. Time series CV

## Hold-out
Simplest and most common technique. The main idea consist in divide your dataset into two: train and test set. Perform the training and the train set and the validation with the test set. Normally the proportion is 80-20.

For this you can use `sklearn.model_selection import train_test_split`

```jupyter
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=111)
```
**Advantages:** It is faster and only requires one iteration
**Disadvantages:** If the data is not uniformly distributed or unbalanced it might happen that the proportion of the training set doesn't contain enough information about the entire dataset and perform poorly on one class. 

## K-fold cross-validation
This algorithm tries to solve the problem that "hold-out" presents, the bottle neck of training once. In the following the steps are depicted:
1. You have the entire dataset, now select a number by which the dataset will be split. For example, if you dataset contain 100 rows and you select K=5, each new slide will contain 20 rows. K received the name of `fold` in this case you will have *5 folds*. Typical values for k are 5 or 10.
2. Select *K-1* folds as the training set, the remaining fold will be the validation set.
3. Train the model on the train set and validate it on the validation set.
4. select a new and different fold to be the validation set, create a new model and perform step 3.
5. repeat step 3 and 4 until you have use all folds as validation set.
6. To get the final score average the results from the K models.

*Observation:* This algorithm is about to make slides of the data, create as many models as slides and train and validate according to each slide. The final score will be the average of all models

you can perform k-fold cv with `from sklear.model_selection import KFold`
```jupyter
import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

In this case we are selecting 2 folds, so the data will be split into 2 slides: one training and one validations: 50-50%, two models will be created changing which is the training and which is the validation set.

**Advantage:** More robust because it considers more varieties on the training and validation 
**Disadvantage:** It creates k-models which is more time consuming. You need to be careful when dataset are small

## Leave-one-out and Leave-p-out cross-validation
The main idea is pretty similar to K-fold but in this case we are using *one* or *p* samples for the test set, and the other samples are for training. So the algorithm will go to the n samples for the case of leave one out of n-p for p out.
Leave-one-out and leave-p-out can be implemented in the following way:
```jupyter
import numpy as np
from sklearn.model_selection import LeaveOneOut

X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```
In this case `test_index` will be just one index and it will vary until all data in the dataset is used a test

```jupyter
import numpy as np
from sklearn.model_selection import LeavePOut

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
lpo = LeavePOut(2)

for train_index, test_index in lpo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```
**Advantage** It uses almost all the data for training
**Disadvantage:** It will make n or n-p models which is expensive


## Stratified k-fold cross validation
This is an extension of K-fold and deals with the problem of having an unbalanced dataset. There are more classes from one type than other. So in the initial k-fold you could take only one class type completely neglecting the other(s) one(s).
In this case `stratifies k-fold CV` makes sure that each fold contains the same proportion of classes than the initial dataset.
In case of numerical values (no classes, linear regression) it assures that the mean target value is approximately equal in all the folds.
The steps are identical as for k-fold.

You can implement it by using `sklearn.model_selection import StratifiedKFold`
```jupyter
import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

**Advantage:** Solves one of the disadvantages for k-fold in the case of unbalanced dataset classes
**disadvantages:** The same from k-fold

## Repeated k-fold cross-validation
*Probably the most robust of all cv* 
In this case you don't specify the number of *folds* but the number of times each fold will be trained with the same model.
==idea== You specify **k** times the model will be trained, the samples will be randomly selected according to the proportion you specified, for example 80-20%.
1. select the number of times it will be trained: k
2. select the number of folds: `n_splits` 
3. Data is split according to the number of folds
4. A different model is trained on each fold
5. validation is perform on the k-1 fold
6. Repeat 3-5 k times. In this case the model for each fold is keep it
7. The final score will be the average of the k performances

So in this case, you have an additional parameter. In K fold you can decide how many "slices" the model will be split, now you can determine this and also how many times each model belonging to each slice will be trained.  One important difference is that now the indices belonging to each fold will be selected randomly

**Advantages:** The number of times a model is trained is independent of the number of folds. It is possible to set different folds to each iteration. It is more robust against bias.
**Disadvantages:** Given that the samples are selected randomly, there is no certainty about that all samples will be selected. It can be the case in which some samples are never used.

You can implement it by using `from sklearn.model_selection import RepeatedKFold`
```jupyter
import numpy as np
from sklearn.model_selection import RepeatedKFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)

for train_index, test_index in rkf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```

## Nested k-Fold
This is different from the previous ones. Until now, you could know how well you model performed for certain problem type and dataset. In this occasion, `Nested K-fold` allow you to search for the best hyperparameter of a model.
It does an hyperparameter search that optimizes the model performance
 Unfortunately, there is no built-in method in sklearn that would perform Nested k-Fold CV for you.
The algorithm of Nested k-Fold technique:

1. Define set of hyper-parameter combinations, C, for current model. If model has no hyper-parameters, C is the empty set.
2. Divide data into K folds with approximately equal distribution of cases and controls.
3. (outer loop) For fold k, in the K folds:
    1. Set fold k, as the test set.
    2. Perform automated feature selection on the remaining K-1 folds.
    3. For parameter combination c in C:
        1. (inner loop) For fold k, in the remaining K-1 folds:
            1. Set fold k, as the validation set.
            2. Train model on remaining K-2 folds.
            3. Evaluate model performance on fold k.
        2. Calculate average performance over K-2 folds for parameter combination c.
    4. Train model on K-1 folds using hyper-parameter combination that yielded best average performance over all steps of the inner loop.
    5.  Evaluate model performance on fold k.
4. Calculate average performance over K folds.