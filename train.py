"""
=============================================================================
  train.py  —  HASTAVEDA Mudra Model Trainer
=============================================================================
  PURPOSE:
      Train a TensorFlow/Keras neural network on the collected mudra landmark
      dataset. Run this script AFTER collect.py has finished.

      Uses the same 47-feature preprocessing pipeline as app.py to ensure
      consistent predictions at runtime.

  HOW TO RUN:
      python train.py

  OUTPUT:
      mudra_data/mudra_model.h5       — Trained Keras model
      mudra_data/mudra_classes.json   — Ordered list of mudra class names
=============================================================================
"""

import os
import sys
import json
import numpy as np

# ── TensorFlow Import ──────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    print(f"[INFO] TensorFlow version: {tf.__version__}")
except ImportError:
    print("[ERROR] TensorFlow is not installed.")
    print("        Run: pip install tensorflow")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "mudra_data")
SAMPLES_PATH = os.path.join(DATASET_PATH, "mudra_samples.json")
MODEL_PATH   = os.path.join(DATASET_PATH, "mudra_model.h5")
CLASSES_PATH = os.path.join(DATASET_PATH, "mudra_classes.json")


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (must match app.py exactly — 47 features)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_landmarks(landmarks):
    """
    Input : list or 1D array of 42 coordinates [x0..x20, y0..y20]
    Output: preprocessed 1D array of 47 features:
      - 21 x-coords centered at wrist, normalized by max distance
      - 21 y-coords centered at wrist, normalized by max distance
      -  5 fingertip-to-wrist distance ratios (thumb, index, middle, ring, pinky)
    """
    x = np.array(landmarks[:21], dtype=np.float32)
    y = np.array(landmarks[21:], dtype=np.float32)

    # 1. Translate wrist (landmark 0) to origin
    x_centered = x - x[0]
    y_centered = y - y[0]

    # 2. Scaling factor — max distance from wrist
    distances = np.sqrt(x_centered ** 2 + y_centered ** 2)
    max_dist  = np.max(distances)
    if max_dist == 0:
        max_dist = 1.0

    # 3. Normalize
    x_norm = x_centered / max_dist
    y_norm = y_centered / max_dist

    # 4. Fingertip-to-wrist distance ratios (5 tips: 4, 8, 12, 16, 20)
    fingertips = [4, 8, 12, 16, 20]
    ft_ratios  = [distances[idx] / max_dist for idx in fingertips]

    return np.concatenate([x_norm, y_norm, ft_ratios])   # shape: (47,)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset():
    if not os.path.exists(SAMPLES_PATH):
        print(f"[ERROR] Dataset file not found: {SAMPLES_PATH}")
        print("        Run collect.py first to gather training data.")
        sys.exit(1)

    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    mudra_classes = list(data.keys())
    features, labels = [], []
    skipped = 0

    for idx, mudra in enumerate(mudra_classes):
        samples = data.get(mudra, [])
        class_count = 0
        for sample in samples:
            if len(sample) == 42:
                features.append(preprocess_landmarks(sample))
                labels.append(idx)
                class_count += 1
            else:
                skipped += 1
        print(f"  [{mudra:15s}]  {class_count} samples loaded")

    if skipped:
        print(f"  [WARNING] {skipped} malformed samples were skipped (expected 42 coords each).")

    features = np.array(features, dtype=np.float32)
    labels   = np.array(labels,   dtype=np.int32)

    print(f"\n  Total samples : {len(features)}")
    print(f"  Feature shape : {features.shape[1]} features per sample")
    print(f"  Classes       : {mudra_classes}")
    return features, labels, mudra_classes


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int, input_dim: int = 47) -> Sequential:
    """
    Three hidden Dense layers with BatchNorm + Dropout for regularization.
    Accepts 47 preprocessed features (matching preprocess_landmarks output).
    """
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation="relu"),
        Dropout(0.1),

        Dense(num_classes, activation="softmax"),
    ], name="hastaveda_mudra_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train():
    print("=" * 65)
    print("  HASTAVEDA  —  Mudra Model Training")
    print("=" * 65)
    print("\n📂  Loading dataset...")

    features, labels, mudra_classes = load_dataset()
    num_classes = len(mudra_classes)

    if len(features) < num_classes * 2:
        print(f"\n[ERROR] Not enough samples to train ({len(features)} total, need at least {num_classes * 2}).")
        print("        Run collect.py to gather more data.")
        sys.exit(1)

    # Train / test split (stratified to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")

    # Build model
    print("\n🏗️   Building model...")
    model = build_model(num_classes=num_classes, input_dim=features.shape[1])
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    print("\n🚀  Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n📊  Evaluation on test set:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test accuracy : {test_acc * 100:.2f}%")
    print(f"  Test loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n  Per-class report:")
    print(classification_report(y_test, y_pred, target_names=mudra_classes))

    # Save model
    print("\n💾  Saving model and labels...")
    os.makedirs(DATASET_PATH, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"  ✅  Model saved   → {MODEL_PATH}")

    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(mudra_classes, f, indent=2)
    print(f"  ✅  Classes saved → {CLASSES_PATH}")

    print("\n" + "=" * 65)
    print(f"  Training complete!  Final accuracy: {test_acc * 100:.2f}%")
    print("  You can now run reduce_photos.py and then app.py.")
    print("=" * 65)


if __name__ == "__main__":
    train()
