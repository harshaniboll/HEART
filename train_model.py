import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("data/heart.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# ==============================
# 2. CHECK MISSING VALUES
# ==============================

print("\nMissing values:")
print(df.isnull().sum())

# Remove missing values if any
df = df.dropna()

print("\nShape after removing missing values:", df.shape)

# ==============================
# 3. SEPARATE FEATURES & TARGET
# ==============================

X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("target")

# ==============================
# 4. SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==============================
# 5. FEATURE SCALING
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 6. TRAIN MODEL
# ==============================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

# ==============================
# 7. PREDICTION
# ==============================

y_pred = model.predict(X_test_scaled)

# ==============================
# 8. MODEL EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 9. SAVE MODEL
# ==============================

os.makedirs("model", exist_ok=True)

model_data = {
    "model": model,
    "scaler": scaler,
    "columns": X.columns.tolist()
}

with open("model/heart_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\n==============================")
print("SUCCESS!")
print("==============================")
print("Model saved successfully!")
print("Location: model/heart_model.pkl")