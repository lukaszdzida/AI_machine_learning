# Common supervised learning models

This document briefly describes commonly used supervised learning models, when to use them, and short notes about strengths and weaknesses.

## Linear regression (for regression)
Linear regression models the relationship between one or more features and a continuous target as a linear function:
- Simple and highly interpretable.
- Assumes linear relationship, constant variance (homoscedasticity), and independence of errors.
- Sensitive to outliers and multicollinearity.
- Variants: ordinary least squares, Ridge (L2), Lasso (L1).

Example (scikit-learn):
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Logistic regression (for classification)
Logistic regression is used for binary (and with extensions, multiclass) classification. It models the probability that a sample belongs to a class using the logistic (sigmoid) function.
- Interpretable coefficients (log-odds).
- Works well for linearly separable problems.
- Probabilistic outputs that are useful for thresholding and calibration.
- Can be regularized (L1, L2) to reduce overfitting.

Example:
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)
```

## Decision trees (for classification and regression)
Decision trees split the feature space into regions using a tree structure.
- Easy to visualize and interpret.
- Can capture nonlinear relationships and feature interactions.
- Prone to overfitting (especially deep trees); control with max depth, min samples, pruning.
- Handles mixed types and missing values (depending on implementation).

## Support vector machines (SVM) (for classification)
SVMs find a decision boundary (hyperplane) that maximizes the margin between classes.
- Effective in high-dimensional spaces.
- Kernel trick lets them handle nonlinear problems (RBF, polynomial kernels).
- Sensitive to choice of kernel and hyperparameters (C, gamma).
- Not ideal for very large datasets (can be slow).

## k-nearest neighbors (k-NN) (for classification and regression)
k-NN predicts based on labels of the k closest training samples in feature space.
- Simple, instance-based (no explicit training phase).
- Works well for low-dimensional data with meaningful distance metrics.
- Computationally expensive at prediction time; sensitive to feature scaling and irrelevant features.

## Random forests (for classification and regression)
Random forests are ensembles of decision trees built with bootstrap samples and random feature selection.
- Robust, good off-the-shelf performance.
- Reduces overfitting compared to single decision trees.
- Provides feature importance measures.
- Less interpretable than a single tree but more accurate for many tasks.

## Neural networks (for classification and regression)
Neural networks consist of layers of interconnected units and can model complex nonlinear functions.
- Extremely flexible: from small feedforward nets to deep convolutional or transformer models.
- Require more data and tuning (architecture, learning rate, regularization).
- Can achieve state-of-the-art results for many domains (images, text, speech).
- Less interpretable; training can be compute-intensive.

## Choosing a model — practical tips
- Start simple: linear models or trees; use them as baseline.
- Use cross-validation for model selection and hyperparameter tuning.
- Preprocess features (scaling, encoding) depending on model (SVM and k-NN need scaling; tree-based models usually do not).
- Consider interpretability vs performance trade-offs.
- Ensemble methods (bagging, boosting) often improve performance if single models underperform.

## References / next steps
- Try simple experiments with scikit-learn to compare models on a dataset.
- Use tools like GridSearchCV / RandomizedSearchCV for hyperparameter tuning.
- For production, add model validation, calibration, and monitoring.
