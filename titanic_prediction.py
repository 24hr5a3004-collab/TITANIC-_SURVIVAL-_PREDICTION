import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (replace with your actual dataset path)
# For demo, we'll create sample data or you can load from URL
# df = pd.read_csv('titanic.csv')

# Sample dataset creation (use this if you don't have the actual dataset)
# In real scenario, download from: https://www.kaggle.com/c/titanic/data
print("Loading Titanic dataset...")

# If you have the actual dataset, uncomment this:
# df = pd.read_csv('titanic.csv')

# For demonstration, let's create a small sample dataset
# (Replace with actual data loading)
data = {
    'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 2, 2],
    'Name': ['Braund', 'Cumings', 'Heikkinen', 'Futrelle', 'Allen', 'Moran', 'McCarthy', 'Palsson', 'Johnson', 'Nasser'],
    'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'male'],
    'Age': [22, 38, 26, 35, 35, 29, 54, 2, 27, 24],
    'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 0],
    'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.08, 11.13, 8.66],
    'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C']
}
df = pd.DataFrame(data)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Preprocessing
def preprocess_titanic(df):
    # Create a copy
    df_processed = df.copy()
    
    # Fill missing values
    df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())
    df_processed['Embarked'] = df_processed['Embarked'].fillna('S')
    
    # Convert categorical variables
    le_sex = LabelEncoder()
    df_processed['Sex'] = le_sex.fit_transform(df_processed['Sex'])
    
    le_embarked = LabelEncoder()
    df_processed['Embarked'] = le_embarked.fit_transform(df_processed['Embarked'])
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df_processed[features]
    y = df_processed['Survived']
    
    return X, y

# Preprocess data
X, y = preprocess_titanic(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTask 1 - Titanic Survival Prediction")
print(f"Accuracy: {accuracy:.2f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)