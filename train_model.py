import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load and clean data
data = pd.read_csv("Salary Data.csv")
data.dropna(inplace=True)
data = pd.get_dummies(data, columns=['Gender', 'Education Level', 'Job Title'], drop_first=True)

# Feature and target separation
X = data.drop('Salary', axis=1)
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained successfully.")

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse}
    print(f"{name}: MAE = {mae:.4f}, MSE = {mse:.4f}")

# Save model and column names
joblib.dump(model, "model.pkl")
# Save model_columns
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")

# Save test_predictions
joblib.dump((X_test, y_test, y_pred), "test_predictions.pkl")

# Optional: Print model performance
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
print("R² Score:", r2_score(y_test, model.predict(X_test)))

print("Model trained")