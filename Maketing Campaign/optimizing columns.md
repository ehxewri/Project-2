Certainly! Here's how you might format the explanations and Python code examples for analyzing the impact of columns on a model into a Markdown document:

```markdown
# Feature Impact Analysis on Models

Analyzing the impact of features in a DataFrame on the performance of a machine learning model involves several methodologies. Below are some common methods for feature selection and importance analysis:

## 1. Feature Importance from Tree-based Models

Tree-based models like Decision Trees and Random Forests calculate feature importances based on the frequency and effectiveness of a feature in splitting data.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_

for i, j in enumerate(importances):
    print(f"Feature {X_train.columns[i]}, Importance: {j}")
```

## 2. Coefficient Analysis in Linear Models

In normalized linear models, coefficients indicate the influence of each feature.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

for i, coef in enumerate(model.coef_[0]):
    print(f"Feature {X_train.columns[i]}, Coefficient: {coef}")
```

## 3. Permutation Importance

Permutation importance measures feature importance by assessing the impact of shuffling each feature on model performance.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
for i in result.importances_mean.argsort()[::-1]:
    if result.importances_mean[i] - 2 * result.importances_std[i] > 0:
        print(f"{X_train.columns[i]}: {result.importances_mean[i]:.3f} +/- {result.importances_std[i]:.3f}")
```

## 4. Recursive Feature Elimination (RFE)

RFE involves fitting a model and recursively removing the least important features.

```python
from sklearn.feature_selection import RFE

model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X_train_scaled, y_train)

for i in range(X_train.shape[1]):
    print(f"Feature {X_train.columns[i]}, Ranking: {rfe.ranking_[i]}")
```

## 5. SHAP Values

SHAP values break down the contribution of each feature to individual predictions.

```python
import shap

model = RandomForestClassifier().fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar")
```

These methods provide insights into which features significantly influence model predictions, aiding in refining and optimizing machine learning models.
```

This Markdown document provides a structured and formatted overview of various methods for analyzing the impact of features on machine learning models, complete with code examples for implementation.