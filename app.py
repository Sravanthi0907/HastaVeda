import os
import json
import uuid
import shutil
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, url_for

# Try to import TensorFlow for the primary .h5 model
TF_AVAILABLE = False
if os.environ.get('SKIP_TENSORFLOW') or (os.environ.get('RENDER') and not os.environ.get('FORCE_TENSORFLOW')):
    print("[INFO] Render/low-memory environment detected. Skipping TensorFlow to prevent OOM. Falling back to scikit-learn MLP Classifier.")
else:
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        print("[Warning] TensorFlow is not installed. Falling back to scikit-learn MLP Classifier for Mudra recognition.")

# Import scikit-learn as a bulletproof fallback for modern environments (like Python 3.14+)
from sklearn.neural_network import MLPClassifier

# Server-side MediaPipe is disabled. Video upload processing has been moved to the client-side.
MP_AVAILABLE = False
mp_hands = None
mp_drawing = None

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 * 1024  # 50 GB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global models, classifiers, and labels
MUDRA_MODEL = None         # Keras H5 Model
FALLBACK_CLASSIFIER = None  # Scikit-learn MLP Model
MUDRA_CLASSES = []

def preprocess_landmarks(landmarks):
    """
    Input: list or 1D array of 42 coordinates: [x0, ..., x20, y0, ..., y20]
    Returns: preprocessed 1D array of 47 coordinates:
      - Translation-invariant: centered at wrist (landmark 0)
      - Scale-invariant: normalized by maximum Euclidean distance from the wrist
      - Fingertip-to-wrist distance ratios (5 features for thumb, index, middle, ring, pinky tips)
    """
    x = np.array(landmarks[:21], dtype=np.float32)
    y = np.array(landmarks[21:], dtype=np.float32)
    
    # 1. Translate wrist (landmark 0) to origin (0, 0)
    x_centered = x - x[0]
    y_centered = y - y[0]
    
    # 2. Calculate scaling factor (max distance from wrist)
    distances = np.sqrt(x_centered**2 + y_centered**2)
    max_dist = np.max(distances)
    
    if max_dist == 0:
        max_dist = 1.0  # Avoid division by zero
        
    # 3. Normalize scale
    x_normalized = x_centered / max_dist
    y_normalized = y_centered / max_dist
    
    # 4. Fingertip-to-wrist distance ratios
    fingertips = [4, 8, 12, 16, 20]
    ft_ratios = [distances[idx] / max_dist for idx in fingertips]
    
    # Return flattened 47 features
    return np.concatenate([x_normalized, y_normalized, ft_ratios])

def load_model_and_labels():
    global MUDRA_MODEL, MUDRA_CLASSES, FALLBACK_CLASSIFIER
    
    classes_path = os.path.join(app.root_path, 'mudra_data', 'mudra_classes.json')
    model_path = os.path.join(app.root_path, 'mudra_data', 'mudra_model.h5')
    samples_path = os.path.join(app.root_path, 'mudra_data', 'mudra_samples.json')
    
    # 1. Load Class Labels
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            MUDRA_CLASSES = json.load(f)
            print(f"[Success] Class labels loaded: {MUDRA_CLASSES}")
    else:
        MUDRA_CLASSES = ["chandrakala", "pataka", "mushti", "shikara", "mrigasirsha", "alapadma", "samyuta"]
        print(f"[Warning] Class file not found, using default fallback: {MUDRA_CLASSES}")

    # 2. Attempt to load TensorFlow H5 model
    if TF_AVAILABLE and os.path.exists(model_path):
        try:
            MUDRA_MODEL = tf.keras.models.load_model(model_path)
            print("[Success] TensorFlow H5 model loaded successfully!")
            return
        except Exception as e:
            print(f"[Error] Error loading TensorFlow model: {e}")
            
    # 3. Fallback: Train a high-performance scikit-learn MLP Neural Network at startup
    if os.path.exists(samples_path):
        try:
            print("[Info] Initializing robust preprocessed scikit-learn Multi-Layer Perceptron neural network...")
            # Open with utf-8-sig to handle BOM written by Windows tools
            with open(samples_path, 'r', encoding='utf-8-sig') as f:
                samples_data = json.load(f)
                
            features = []
            labels = []
            
            for idx, mudra in enumerate(MUDRA_CLASSES):
                if mudra in samples_data:
                    for sample in samples_data[mudra]:
                        if len(sample) == 42:  # 21 landmarks * 2 coordinates
                            features.append(preprocess_landmarks(sample))
                            labels.append(idx)
                            
            features = np.array(features)
            labels = np.array(labels)
            
            # Instantiate an MLP with optimal layers and hyperparameters
            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                max_iter=300,
                random_state=42
            )
            clf.fit(features, labels)
            FALLBACK_CLASSIFIER = clf
            print(f"[Success] Robust Scikit-learn MLP Neural Network successfully trained on {len(features)} samples!")
        except Exception as e:
            print(f"[Error] Failed to initialize robust fallback classifier: {e}")
    else:
        print(f"[Critical Error] mudra_samples.json not found at {samples_path}")

def copy_mudra_images():
    """
    Extracts one actual image for each mudra directly from the dataset folders and
    saves them to static/images/mudras/ so the flashcards display the real dataset images.
    """
    dest_dir = os.path.join(app.root_path, 'static', 'images', 'mudras')
    os.makedirs(dest_dir, exist_ok=True)
    
    for mudra in MUDRA_CLASSES:
        src_image_path = os.path.join(app.root_path, 'mudra_data', mudra, f"{mudra}_0.jpg")
        dest_image_path = os.path.join(dest_dir, f"{mudra}.jpg")
        if os.path.exists(src_image_path):
            try:
                shutil.copy(src_image_path, dest_image_path)
                print(f"[Success] Dynamic Setup: Copied {mudra}_0.jpg to static/images/mudras/{mudra}.jpg")
            except Exception as e:
                print(f"[Error] Error copying {mudra} image: {e}")

# Initialize backend models and static image copies immediately
load_model_and_labels()
copy_mudra_images()

# ----------------- PAGE ROUTES -----------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detection.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

# ----------------- API ENDPOINTS -----------------

@app.route('/api/predict-landmarks', methods=['POST'])
def predict_landmarks():
    """
    Accepts 42 hand landmarks as JSON array [x0, x1, ... x20, y0, y1, ... y20].
    Coordinates are normalized to a standard 640x480 scale.
    """
    if MUDRA_MODEL is None and FALLBACK_CLASSIFIER is None:
        return jsonify({"error": "No classification model is currently loaded"}), 500
        
    try:
        data = request.json
        if not data or 'landmarks' not in data:
            return jsonify({"error": "Invalid payload"}), 400
            
        landmarks = data['landmarks']
        if len(landmarks) != 42:
            return jsonify({"error": f"Invalid landmark dimensions, expected 42 but got {len(landmarks)}"}), 400
            
        # Both backends use the same 47-feature preprocessed input for consistency.
        # preprocess_landmarks() applies wrist-origin centering, scale normalization,
        # and fingertip-to-wrist ratios — matching train.py exactly.
        preprocessed_features = preprocess_landmarks(landmarks)
        landmarks_array = np.array(preprocessed_features, dtype=np.float32).reshape(1, -1)

        if FALLBACK_CLASSIFIER is not None:
            probabilities = FALLBACK_CLASSIFIER.predict_proba(landmarks_array)[0]
        elif MUDRA_MODEL is not None:
            prediction = MUDRA_MODEL.predict(landmarks_array, verbose=0)
            probabilities = prediction[0]
            
        confidence = float(np.max(probabilities))
        predicted_idx = np.argmax(probabilities)
        predicted_class = MUDRA_CLASSES[predicted_idx]
        
        # Recognition threshold (e.g. 0.85)
        status = "Recognized" if confidence > 0.85 else "Low Confidence"
        
        return jsonify({
            "mudra": predicted_class,
            "confidence": confidence,
            "status": status,
            "confidences": {MUDRA_CLASSES[i]: float(probabilities[i]) for i in range(len(MUDRA_CLASSES))}
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """
    Deprecated: Video processing is now handled entirely client-side.
    This endpoint returns a message pointing users/clients to the client-side system.
    """
    return jsonify({
        "error": "This API endpoint is deprecated. Video processing is now done entirely client-side in the browser."
    }), 400

def secure_filename_uuid(filename):
    """Utility to generate a unique safe filename with original extension"""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.mp4', '.avi', '.mov']:
        ext = '.mp4'
    return f"{uuid.uuid4().hex}{ext}"

if __name__ == '__main__':
    # Start dev server on local port 5000
    app.run(debug=True, port=5000)
