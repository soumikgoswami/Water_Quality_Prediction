import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('PB_All_2000_2021.csv', delimiter=';')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.drop(['date', 'id'], axis=1)
feature_columns = ['year', 'month']
target_columns = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
X = df[feature_columns]
y = df[target_columns]
df_cleaned = pd.concat([X, y], axis=1).dropna(subset=target_columns)
X_cleaned = df_cleaned[feature_columns]
y_cleaned = df_cleaned[target_columns]
print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape (after dropping NaNs in target columns): {df_cleaned.shape}")
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
print("\nFirst 5 Actual vs Predicted values:")
print("Actual:")
print(y_test.head())
print("\nPredicted:")
print(pd.DataFrame(y_pred, columns=y_test.columns).head())