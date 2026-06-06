# HastaVeda — AI-Powered Bharatanatyam Mudra Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9–3.11-blue?logo=python" alt="Python 3.9–3.11"/>
  <img src="https://img.shields.io/badge/Flask-3.x-black?logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.13–2.15-orange?logo=tensorflow" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10-green" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/Render-Ready-46E3B7?logo=render" alt="Render"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="MIT License"/>
</p>

**HastaVeda** is a culturally rich, interactive digital studio for classical Indian dancers. It bridges **ancient Indian dance heritage** (codified in the *Abhinaya Darpana*) with **modern deep neural networks** to provide real-time recognition and learning of Bharatanatyam Hasta Mudras (hand gestures).

---

## Key Features

| Feature | Description |
|---|---|
| **Live Webcam Practice** | Ultra-low latency real-time mudra recognition via client-side MediaPipe + Flask API |
| **Recorded Video Analysis** | Frame-by-frame analysis of uploaded `.mp4`, `.avi`, `.mov` files |
| **Mudra Library** | 3D flip-card dictionary with Sanskrit names, symbolism, and technique notes |
| **Dual-Backend Inference** | TensorFlow `.h5` model (primary) + scikit-learn MLP fallback (automatic) |
| **Cloud Ready** | Configured for headless Render deployment with Gunicorn |

---

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, FontAwesome 6
- **Backend**: Flask (Python 3.9–3.11), Gunicorn (WSGI)
- **Computer Vision**: OpenCV (headless), MediaPipe Hands
- **Machine Learning**: TensorFlow 2.x (training + inference), scikit-learn MLP (fallback)

> **Python version note**: TensorFlow and MediaPipe require **Python 3.9–3.11**. On Python 3.12+, the app runs automatically with the scikit-learn MLP fallback (equally accurate) and server-side video upload is disabled; live webcam remains fully functional.

---

## Project Structure

```
HASTAVEDA/
│
├── collect.py              ← STEP 1: Capture mudra images via webcam
├── train.py                ← STEP 2: Train the neural network model
├── reduce_photos.py        ← STEP 3: Trim dataset for GitHub (run after training)
├── app.py                  ← STEP 4: Run the Flask web application
│
├── requirements.txt        ← Python dependencies
├── Procfile                ← Gunicorn start command (Render/Heroku)
├── render.yaml             ← Render one-click deploy config
├── runtime.txt             ← Python 3.11.9 pin for Render
├── .gitignore
└── README.md
│
├── mudra_data/
│   ├── chandrakala/        ← 10 sample images per mudra (post-reduction)
│   ├── pataka/
│   ├── mushti/
│   ├── shikara/
│   ├── mrigasirsha/
│   ├── alapadma/
│   ├── samyuta/
│   ├── mudra_model.h5      ← Trained Keras model (committed to git)
│   ├── mudra_classes.json  ← Ordered class names list
│   └── mudra_samples.json  ← Landmark coordinates (10 per mudra post-reduction)
│
├── static/
│   ├── css/style.css
│   ├── js/
│   │   ├── main.js         ← UI animations and navigation
│   │   ├── webcam.js       ← Client-side MediaPipe + API integration
│   │   └── upload.js       ← Video upload and timeline player
│   ├── images/mudras/      ← Auto-generated preview images (at startup)
│   └── uploads/.gitkeep    ← Keeps folder tracked; actual uploads are gitignored
│
└── templates/
    ├── base.html           ← Page shell, navbar
    ├── index.html          ← Homepage
    ├── detection.html      ← Live webcam + video upload studio
    └── learn.html          ← Mudra flashcard library
```

---

## Installation

### Prerequisites
- **Python 3.9 – 3.11** — required for TensorFlow 2.x and MediaPipe
- **Webcam** — required for `collect.py` and live detection
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/HASTAVEDA.git
cd HASTAVEDA
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Windows users**: If `mediapipe` fails, confirm you are using Python 3.9–3.11.
> Run `python --version` to verify. You can install a specific version via [python.org](https://www.python.org/downloads/).

---

## Complete Workflow (from scratch)

Follow these steps **in order** to go from a fresh clone to a fully trained application.

---

### Step 1 — Collect Training Data (`collect.py`)

This script opens your webcam and captures **400 images per mudra** (7 mudras = 2,800 images total).

```bash
python collect.py
```

**Controls during collection:**
| Key | Action |
|---|---|
| `Q` | Skip current mudra (saves progress, moves to next) |
| `ESC` | Save all collected data and quit |

**What it does:**
- Saves annotated JPEG images to `mudra_data/<mudra>/`
- Saves landmark coordinates to `mudra_data/mudra_samples.json`
- Saves the class list to `mudra_data/mudra_classes.json`
- ✅ **Resume support**: If interrupted, re-running picks up where it left off

**Tips:**
- Ensure good, consistent lighting
- Keep your hand clearly visible and centred in frame
- Move your hand slightly between captures for variety

---

### Step 2 — Train the Model (`train.py`)

After collecting data, train the neural network:

```bash
python train.py
```

**What it does:**
- Loads `mudra_samples.json` and applies **47-feature preprocessing** (wrist-origin centering + scale normalization + fingertip ratios)
- Trains a 4-layer Dense neural network with Dropout + BatchNorm
- Uses `EarlyStopping` to prevent overfitting
- Prints a per-class accuracy report
- Saves:
  - `mudra_data/mudra_model.h5` — the trained Keras model
  - `mudra_data/mudra_classes.json` — ordered class names

> ⚠️ Training requires TensorFlow (Python 3.9–3.11). On a CPU, training 2,800 samples for up to 100 epochs takes approximately 2–5 minutes.

---

### Step 3 — Reduce Dataset for GitHub (`reduce_photos.py`)

Run this **after training is complete** to shrink the image dataset from 400 → 10 images per mudra so you can push to GitHub without hitting size limits.

```bash
python reduce_photos.py
```

**This script:**
- Keeps only the first 10 images per mudra (photos `0–9`)
- Trims `mudra_samples.json` to 10 samples per mudra
- Leaves `mudra_model.h5` and `mudra_classes.json` completely untouched
- Asks for a `yes` confirmation before deleting anything
- Supports `DRY_RUN = True` (preview mode — no files deleted)

> ✅ The trained model is **not affected** by this reduction. Predictions remain identical.

---

### Step 4 — Run the Application (`app.py`)

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 🪘 Supported Mudras

| Mudra | Sanskrit | Meaning |
|---|---|---|
| Pataka | पताका | Flag |
| Chandrakala | चन्द्रकला | Crescent Moon |
| Mushti | मुष्टि | Fist |
| Shikara | शिखर | Peak / Arrow |
| Mrigasirsha | मृगशीर्ष | Deer's Head |
| Alapadma | अलपद्म | Fully Bloomed Lotus |
| Samyuta | संयुत | Combined (both hands) |

---