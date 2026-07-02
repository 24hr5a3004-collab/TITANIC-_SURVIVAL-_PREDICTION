import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ==================== 1. LOAD DATA ====================
print("="*50)
print("IRIS FLOWER CLASSIFICATION")
print("="*50)

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# ==================== 2. EXPLORE DATA ====================
print("\n📊 DATASET INFORMATION:")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(feature_names)}")
print(f"Features: {feature_names}")
print(f"Target classes: {target_names}")

print("\n📋 First 5 rows:")
print(df.head())

print("\n📈 Statistical Summary:")
print(df[feature_names].describe())

print("\n🎯 Class Distribution:")
print(df['species_name'].value_counts())

# ==================== 3. VISUALIZE DATA ====================
# Pairplot to visualize relationships
sns.pairplot(df, hue='species_name', diag_kind='hist')
plt.suptitle('Iris Dataset - Pairplot', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
correlation = df[feature_names].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Box plots for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feature in enumerate(feature_names):
    row = i // 2
    col = i % 2
    sns.boxplot(data=df, x='species_name', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# ==================== 4. PREPROCESS DATA ====================
print("\n🔄 PREPROCESSING DATA...")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed!")

# ==================== 5. TRAIN MODELS ====================
print("\n🤖 TRAINING MODELS...")

# Dictionary to store models and their accuracies
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# ==================== 6. EVALUATE BEST MODEL ====================
# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']
best_predictions = results[best_model_name]['predictions']

print("\n" + "="*50)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"Accuracy: {best_accuracy * 100:.2f}%")
print("="*50)

# ==================== 7. CONFUSION MATRIX ====================
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== 8. CLASSIFICATION REPORT ====================
print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, best_predictions, target_names=target_names))

# ==================== 9. FEATURE IMPORTANCE ====================
# Only for Random Forest
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🌟 FEATURE IMPORTANCE:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.show()

# ==================== 10. TEST WITH NEW DATA ====================
print("\n🔮 TESTING WITH NEW SAMPLES:")

# Create 3 new test samples
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Should be setosa
    [6.0, 3.0, 4.8, 1.8],  # Should be versicolor or virginica
    [6.5, 3.0, 5.2, 2.0]   # Should be virginica
])

# Scale the new samples
new_samples_scaled = scaler.transform(new_samples)

# Predict
predictions = best_model.predict(new_samples_scaled)
prediction_proba = best_model.predict_proba(new_samples_scaled)

print("\nSample Predictions:")
for i, (sample, pred, proba) in enumerate(zip(new_samples, predictions, prediction_proba)):
    print(f"\nSample {i+1}: {sample}")
    print(f"Predicted: {target_names[pred]}")
    print(f"Confidence: {np.max(proba) * 100:.2f}%")
    print(f"All probabilities: Setosa: {proba[0]*100:.1f}%, "
          f"Versicolor: {proba[1]*100:.1f}%, "
          f"Virginica: {proba[2]*100:.1f}%")

# ==================== 11. SAVE MODEL ====================
import joblib

# Save the best model and scaler
joblib.dump(best_model, 'iris_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')
print("\n💾 Model saved as 'iris_model.pkl' and 'iris_scaler.pkl'")

# ==================== 12. LOAD AND USE MODEL ====================
# Example of how to load and use the saved model
print("\n📂 LOADING SAVED MODEL...")

loaded_model = joblib.load('iris_model.pkl')
loaded_scaler = joblib.load('iris_scaler.pkl')

test_sample = np.array([[5.0, 3.4, 1.5, 0.2]])
test_sample_scaled = loaded_scaler.transform(test_sample)
prediction = loaded_model.predict(test_sample_scaled)

print(f"Loaded model prediction: {target_names[prediction[0]]}")

print("\n✅ Task 3 Completed Successfully!")