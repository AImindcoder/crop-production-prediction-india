# crop_prediction.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Step 1: Load the dataset
# -----------------------
csv_path = r"C:\Users\Sahil khan(Machine E\Music\New folder (2)\crop_production.csv"
df = pd.read_csv(csv_path)

# Display basic info
print("Dataset Info:")
print(df.info())

# -----------------------
# Step 2: Preprocessing
# -----------------------
# Fill missing numeric values with 0
df['Area'] = df['Area'].fillna(0)
df['Production'] = df['Production'].fillna(0)

# Encode categorical columns using dummy variables
df_encoded = pd.get_dummies(df, columns=['State_Name', 'District_Name', 'Season', 'Crop'])

# -----------------------
# Step 3: Feature selection
# -----------------------
# Use all columns except Production as features
X = df_encoded.drop(['Production'], axis=1)
y = df_encoded['Production']

# -----------------------
# Step 4: Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# Step 5: Train a Linear Regression model
# -----------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------
# Step 6: Predictions
# -----------------------
y_pred = model.predict(X_test)

# -----------------------
# Step 7: Model evaluation
# -----------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# -----------------------
# Step 8: Plot actual vs predicted
# -----------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Crop Production")
plt.show()

# -----------------------
# Step 9: Predict production for a new sample (example)
# -----------------------
# Example: Replace with real values
new_sample = X_test.iloc[0].values.reshape(1, -1)
predicted_value = model.predict(new_sample)
print(f"\nPredicted production for the sample: {predicted_value[0]:.2f}")
