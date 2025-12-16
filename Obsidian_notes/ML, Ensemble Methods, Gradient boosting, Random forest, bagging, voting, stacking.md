## Reference
 https://scikit-learn.org/stable/modules/ensemble.html#id42


**Ensemble methods** combine the prediction of several models to give one and more robust prediction than the one that will be obtained by each individual model.

## Gradient-boosted trees (GBT)

GBT is an excellent model for both regression and classification, *in particular for tabular data*

### Difference between GBT and RandomForest:
Both models represent ensembles of decision trees but they differ in *training process* and *how they combine the individual tree's outputs*.

**Main difference** is that RandomForest are an ensemble of trees that are trained independently and on different subsets of features (parallel trees). Whereas the ***GBT*** trains the trees in series. It means that it trains a first tree, make the prediction and compute the error between the prediction and the ground truth. Then the second tree will be trained on the residual error of the first tree to improve this, so we calculate the new residual. The third three will be trained on this new residual and so on until the residual reaches a minimum value.
### Implementation with Scikit-learn:
Scikit-learn provide two versions: [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier "sklearn.ensemble.GradientBoostingClassifier") and [`HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier "sklearn.ensemble.HistGradientBoostingClassifier") 
* HistGradientBoostingClassifier: it is several orders of magnitude faster and it is preferred for *larger data (tens of thousands of data)* . Additionally, missing values and categorical data are natively support, removing the need for additional imputation
* GradientBoostingClassifier: it is preferred for *small* datasets.

## Bagging meta-estimator
The basic idea is to select any estimator you want to use and the approach about how the samples or the features will be used is different. For example you select an estimator "A" (decision tree, k-means, linear regression, etc), then the model will build several instances of the estimator and it will select the data in different ways:
* random subsets of the data are drawn as *random subsets of the samples* **Pasting**
* when samples are drawn with *replacement* **Bagging**
* when random subsets of the dataset are drawn as *random subsets of the features* **Random subspaces**
* when both *samples and features subsets* are selected **Random patches**

