import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic sample data instead of downloading
np.random.seed(42)
n_samples = 1000
X = pd.DataFrame({
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'feature_4': np.random.randn(n_samples),
    'feature_5': np.random.randn(n_samples),
    'feature_6': np.random.randn(n_samples),
    'feature_7': np.random.randn(n_samples),
    'feature_8': np.random.randn(n_samples),
})
y = pd.Series(X['feature_1'] * 2 + X['feature_2'] * 1.5 + np.random.randn(n_samples) * 0.1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

outlier_features = []

for col in X_train.columns:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = X_train[(X_train[col] < lower) | (X_train[col] > upper)]
    if len(outliers) > 0:
        outlier_features.append(col)

features = outlier_features
