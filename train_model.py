import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

X = np.load(
    "X.npy"
)

y = np.load(
    "y.npy"
)


print()
print("Dataset loaded.")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print()
print("Training samples:",
      len(X_train))

print("Testing samples:",
      len(X_test))


# ============================================================
# ML PIPELINE
# ============================================================

model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",

        RandomForestClassifier(

            n_estimators=200,

            random_state=42,

            class_weight="balanced",

            n_jobs=1
        )
    )
])


# ============================================================
# TRAIN
# ============================================================

print()
print("Training Random Forest...")
print()

model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST
# ============================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print()
print("======================================")
print("MODEL RESULTS")
print("======================================")

print()

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


print()
print("Classification report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Noise",
            "Speech"
        ]
    )
)


print()
print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "speech_noise_model.pkl"
)


print()
print("======================================")

print(
    "Model saved as:"
)

print(
    "speech_noise_model.pkl"
)

print("======================================")