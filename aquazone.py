
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("Loading AquaZone dataset...")
data = pd.read_csv(r"C:\Users\Ranjith\Downloads\aquazone_dataset_2000_rows.csv")

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)
print("\nFirst 5 rows of dataset:\n", data.head())


print("\nChecking for missing values...")
print(data.isnull().sum())

features = [
    'Condenser_Temp',
    'Ambient_Temp',
    'Humidity',
    'Pump_Status'
]

target = 'Power_Consumption'

X = data[features]
y = data[target]

print("\nSelected Features:\n", X.columns)
print("Target Variable:", target)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)
print("Model training completed")


y_pred = rf_model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics")
print("------------------------")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Power Consumption (W)")
plt.ylabel("Predicted Power Consumption (W)")
plt.title("Random Forest Model: Actual vs Predicted Power Consumption")
plt.grid(True)
plt.show()

residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Power Consumption (W)")
plt.ylabel("Residual Error (W)")
plt.title("Residual Analysis of Random Forest Model")
plt.grid(True)
plt.show()


importances = rf_model.feature_importances_

plt.figure()
plt.bar(features, importances)
plt.xlabel("Input Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Power Consumption Prediction")
plt.grid(True)
plt.show()

sample_input = np.array([[35, 32, 60, 1]])
sample_input_scaled = scaler.transform(sample_input)

predicted_power = rf_model.predict(sample_input_scaled)

print("\nSample Input:")
print("Condenser Temp = 35°C, Ambient Temp = 32°C, Humidity = 60%, Pump = ON")
print("Predicted Power Consumption:", predicted_power[0], "Watts")
