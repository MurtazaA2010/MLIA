# AnLOF

**AnLOF** is a machine learning utility library built to automatically handle outliers effectively.

AnLOF uses multiple outlier-handling techniques and automatically selects the best-performing method.

---

## Outlier Handling Methods Used

- IQR Score
- Z-score
- Winsorization
- Median Imputation
- Mean Imputation
- Isolation Forest
- Box-Cox Transformation
- k-Nearest Neighbors
- XGBRegressor
- LGBMRegressor
- CatBoostRegressor
- StandardScaler
- RobustScaler
- MinMaxScaler
- Log Transformation
- Quantile Normalization

---

## Features

- Automatic outlier detection and handling
- Multiple preprocessing strategies
- Model-based evaluation
- Selects the best method based on the chosen metric
- Returns your dataset with the best preprocessing method applied

---

## Hyperparameters

- **X_train** : Your training feature set  
- **X_val** : Your validation feature set  
- **y_train** : Your training target values  
- **y_val** : Your validation target values  

- **features** : Features that contain outliers  
  > Note: This should NOT include all features in your dataset, only those suspected of containing outliers.

- **base_model** : The model used to evaluate the performance of each preprocessing method.

- **metric** : The evaluation metric used to compare performance.

- **higher_is_better** :  
  - `True` for metrics where higher values are better (e.g. `accuracy_score`, `f1_score`)  
  - `False` for metrics where lower values are better (e.g. `mean_squared_error`)

---


## Author

Author

This project is developed and maintained by MurtazaA2010.

For any questions or support:
Email : murtazaabdullah989@gmail.com
Web : [Murtaza Abdullah](https://murtazaadbullah10.web.app)


## Installation

```bash
pip install AnLOF

```python
from AnLOF.AnLOF_module import AnLOF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

anlof = AnLOF(
    X_train,
    X_val,
    y_train,
    y_val,
    features=["feature1", "feature2"],  # features containing outliers
    base_model=LinearRegression,
    metric=mean_squared_error,
    higher_is_better=False
)

best_X_train, best_X_val, best_method, performance_df = anlof.forward()

# best_method : the method with the best score
# performance_df : contains the performance of all the methods

print("Best method:", best_method)

print("Best X_train:")
print(best_X_train.head())

print("Best X_val:")
print(best_X_val.head())

print("Performance comparison:")
print(performance_df)
