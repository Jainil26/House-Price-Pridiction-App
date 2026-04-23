from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


data = fetch_california_housing()

df = pd.DataFrame(data.data , columns=data.feature_names)
df['Price'] = data.target

# print(df.shape)
# print(df.head())

# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(),annot=True,cmap='coolwarm',fmt='.2f')
# plt.title('Feature Correlation')
# plt.show()

# sns.histplot(df['Price'], bins=50)
# plt.title('House Price Distribution')
# plt.show()

x = df.drop('Price',axis=1)
y = df['Price']

x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)

print("MAE: ", round(mean_absolute_error(y_test, y_pred_lr), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 3))
print("R²:  ", round(r2_score(y_test, y_pred_lr), 3))

rf = RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)

print("--- Random Forest ---")
print("MAE: ", round(mean_absolute_error(y_test, y_pred_rf), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 3))
print("R²:  ", round(r2_score(y_test, y_pred_rf), 3))

# importances = pd.Series(rf.feature_importances_, index=data.feature_names)
# importances.sort_values().plot(kind='barh', title='Feature Importance')
# plt.show()

joblib.dump(rf, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Saved successfully!")