import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create sample data
np.random.seed(42)
n_samples = 1000

sample_data = {
    'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'], n_samples),
    'director': np.random.choice(['Director A', 'Director B', 'Director C'], n_samples),
    'actors': np.random.choice(['Actor 1', 'Actor 2', 'Actor 3', 'Actor 4'], n_samples),
    'year': np.random.randint(1990, 2023, n_samples),
    'rating': np.random.uniform(1, 10, n_samples)
}

df = pd.DataFrame(sample_data)
print("Sample dataset created!")

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_cols = ['genre', 'director', 'actors']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('rating', axis=1)
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")