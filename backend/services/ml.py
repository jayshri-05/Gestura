from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "model.joblib"
FEATURES = 21 * 3

def validate_landmarks(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (FEATURES,) or not np.isfinite(arr).all():
        raise ValueError(f"Expected {FEATURES} finite landmark coordinates")
    return arr

def train(rows):
    if len(rows) < 4:
        raise ValueError("Collect at least four samples before training")
    labels = sorted({r["label"] for r in rows})
    if len(labels) < 2:
        raise ValueError("Training requires samples from at least two gesture labels")
    X = np.stack([validate_landmarks(r["landmarks"]) for r in rows])
    y = np.asarray([r["label"] for r in rows])
    counts = {label: int((y == label).sum()) for label in labels}
    if min(counts.values()) < 2:
        raise ValueError("Each label needs at least two samples for a held-out evaluation")
    test_count = max(len(labels), int(round(len(y) * .2)))
    if test_count >= len(y):
        test_count = len(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_count, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    train_acc = float(accuracy_score(y_train, model.predict(X_train)))
    test_acc = float(accuracy_score(y_test, model.predict(X_test)))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return {"training_accuracy": train_acc, "testing_accuracy": test_acc,
            "sample_count": len(rows), "class_count": len(labels), "per_label": counts}

def predict(landmarks):
    if not MODEL_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    vector = validate_landmarks(landmarks).reshape(1, -1)
    probs = model.predict_proba(vector)[0]
    idx = int(np.argmax(probs))
    return {"label": str(model.classes_[idx]), "confidence": float(probs[idx])}
