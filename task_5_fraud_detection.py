import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("CREDIT CARD FRAUD DETECTION")
print("="*50)

# ==================== 1. CREATE SAMPLE DATA ====================
# If you have actual dataset, replace this with:
# df = pd.read_csv('creditcard.csv')

np.random.seed(42)
n_samples = 5000
n_features = 10

# Create normal transactions (98% of data)
normal = int(n_samples * 0.98)
fraud = n_samples - normal

# Generate features
X_normal = np.random.normal(0, 1, (normal, n_features))
X_fraud = np.random.normal(2, 1.5, (fraud, n_features))

X = np.vstack([X_normal, X_fraud])
y = np.hstack([np.zeros(normal), np.ones(fraud)])

# Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['is_fraud'] = y

print("\n📊 DATASET INFO:")
print(f"Total transactions: {len(df)}")
print(f"Normal: {len(df[df['is_fraud']==0])}")
print(f"Fraud: {len(df[df['is_fraud']==1])}")
print(f"Fraud percentage: {len(df[df['is_fraud']==1])/len(df)*100:.2f}%")

# ==================== 2. PREPARE DATA ====================
# Features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")

# ==================== 3. TRAIN MODELS ====================
print("\n🤖 TRAINING MODELS...")

# Model 1: Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# ==================== 4. EVALUATE MODELS ====================
print("\n📊 RESULTS:")
print("-"*50)

# Function to calculate metrics
def evaluate(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    [[{cm[0][0]:4d} {cm[0][1]:4d}]")
    print(f"     [{cm[1][0]:4d} {cm[1][1]:4d}]]")

# Evaluate both models
evaluate(y_test, y_pred_lr, "Logistic Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# ==================== 5. BEST MODEL ====================
# Choose best model based on F1-score (balances precision and recall)
f1_lr = f1_score(y_test, y_pred_lr)
f1_rf = f1_score(y_test, y_pred_rf)

if f1_rf > f1_lr:
    best_model = rf
    best_name = "Random Forest"
    best_pred = y_pred_rf
else:
    best_model = lr
    best_name = "Logistic Regression"
    best_pred = y_pred_lr

print("\n" + "="*50)
print(f"🏆 BEST MODEL: {best_name}")
print("="*50)

# ==================== 6. TEST WITH NEW TRANSACTION ====================
print("\n🔮 TEST WITH NEW TRANSACTION:")

# Create a new transaction
new_transaction = np.random.normal(0, 1, (1, n_features))
new_transaction_scaled = scaler.transform(new_transaction)

# Predict
prediction = best_model.predict(new_transaction_scaled)
probability = best_model.predict_proba(new_transaction_scaled)

if prediction[0] == 0:
    status = "✅ GENUINE"
else:
    status = "⚠️  FRAUD"

print(f"Transaction: {new_transaction[0][:5]}...")
print(f"Prediction: {status}")
print(f"Fraud Probability: {probability[0][1]*100:.2f}%")
print(f"Genuine Probability: {probability[0][0]*100:.2f}%")

# ==================== 7. SAVE MODEL (Optional) ====================
import joblib

joblib.dump(best_model, 'fraud_model.pkl')
joblib.dump(scaler, 'fraud_scaler.pkl')
print("\n💾 Model saved as 'fraud_model.pkl'")

print("\n✅ Task 5 Complete!")