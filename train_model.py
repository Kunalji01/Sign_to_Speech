import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset folder
dataset_path = "dataset"

X = []
y = []

# Signs to train
signs = [
    os.path.splitext(file)[0]
    for file in os.listdir(dataset_path)
    if file.endswith(".csv")
]

print("Signs found:", signs)

# Load each CSV file
for sign in signs:
    file_path = os.path.join(dataset_path, f"{sign}.csv")

    data = pd.read_csv(file_path, header=None)

    print(f"{sign.upper()}: {len(data)} samples loaded")

    # Features
    X.extend(data.values)

    # Labels
    y.extend([sign] * len(data))

print("\nTotal samples:", len(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/sign_model.pkl")

print("\nModel saved successfully!")