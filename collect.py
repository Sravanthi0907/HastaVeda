"""
=============================================================================
  collect.py  —  HASTAVEDA Mudra Dataset Collector
=============================================================================
  PURPOSE:
      Capture 400 images + landmark samples for each mudra using your webcam.
      Run this script BEFORE train.py.

      If collection is interrupted, re-running the script will RESUME from
      where it left off — existing images are not overwritten.

  CONTROLS (while the webcam window is open):
      Q   — Skip to next mudra (saves progress so far)
      ESC — Quit and save all collected data immediately

  HOW TO RUN:
      python collect.py

  OUTPUT:
      mudra_data/<mudra>/        — Annotated JPEG images
      mudra_data/mudra_samples.json  — Landmark coordinates for training
      mudra_data/mudra_classes.json  — Ordered class name list
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta

# ── MediaPipe Setup ────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # One hand per frame — cleaner data
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6,
)

# ── Dataset Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "mudra_data")
SAMPLES_PATH = os.path.join(DATASET_PATH, "mudra_samples.json")
CLASSES_PATH = os.path.join(DATASET_PATH, "mudra_classes.json")
os.makedirs(DATASET_PATH, exist_ok=True)

# ── Mudra Classes ─────────────────────────────────────────────────────────────
MUDRAS = ["chandrakala", "pataka", "mushti", "shikara", "mrigasirsha", "alapadma", "samyuta"]

# ── Collection Settings ───────────────────────────────────────────────────────
TOTAL_SAMPLES_PER_MUDRA = 400      # Target images per mudra
SAMPLES_PER_BREAK       = 100      # Rest every N images
CAPTURE_DELAY           = 0.08     # Seconds between captures (80 ms = ~12 fps)
POSE_PREP_TIME          = 5        # Seconds to prepare pose before capture starts
BREAK_TIME_SMALL        = 15       # Seconds of rest after every 100 images
BREAK_TIME_LARGE        = 10       # Seconds of rest after finishing 400 images


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_data() -> dict:
    """Load previously collected landmark samples so collection can resume."""
    if os.path.exists(SAMPLES_PATH):
        try:
            with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {mudra: [] for mudra in MUDRAS}


def count_existing_images(mudra: str) -> int:
    """Count how many images already exist in a mudra folder."""
    folder = os.path.join(DATASET_PATH, mudra)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])


def save_data(data: dict):
    """Persist landmark samples to JSON."""
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def save_classes():
    """Persist the ordered list of mudra class names."""
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(MUDRAS, f, indent=2)


def save_annotated_image(frame, mudra: str, count: int):
    """Save a JPEG frame to the mudra folder."""
    folder = os.path.join(DATASET_PATH, mudra)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{mudra}_{count}.jpg")
    cv2.imwrite(path, frame)


def extract_landmarks(hand_landmarks, width: int, height: int):
    """
    Extract scaled (x, y) landmark coordinates from a MediaPipe hand result.
    Returns (landmarks_42, bbox) where:
      landmarks_42 = [x0..x20, y0..y20]  (pixel-scaled)
      bbox         = (x_min, y_min, x_max, y_max)
    """
    x_coords = [lm.x * width  for lm in hand_landmarks.landmark]
    y_coords = [lm.y * height for lm in hand_landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    return x_coords + y_coords, (x_min, y_min, x_max, y_max)


def draw_bounding_box(frame, bbox):
    x_min, y_min, x_max, y_max = bbox
    # Expand box slightly for visual clarity
    pad = 15
    cv2.rectangle(
        frame,
        (max(0, x_min - pad), max(0, y_min - pad)),
        (x_max + pad, y_max + pad),
        (0, 255, 120), 2
    )


def draw_overlay(frame, mudra: str, count: int, total: int, extra_text: str = ""):
    """Draw HUD overlay on the frame."""
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Mudra: {mudra.upper()}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2)
    cv2.putText(frame, f"Captured: {count}/{total}", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Progress bar
    bar_w = w - 20
    filled = int(bar_w * count / total)
    cv2.rectangle(frame, (10, 80), (10 + bar_w, 100), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 80), (10 + filled, 100), (0, 220, 100), -1)

    if extra_text:
        cv2.putText(frame, extra_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.putText(frame, "Q=Skip  ESC=Quit", (w - 220, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def countdown(cap, title: str, seconds: int, subtitle: str = ""):
    """Display a full-screen countdown before capture begins."""
    end_time = datetime.now() + timedelta(seconds=seconds)
    while datetime.now() < end_time:
        ret, frame = cap.read()
        if not ret:
            break
        secs_left = int((end_time - datetime.now()).total_seconds()) + 1
        h, w = frame.shape[:2]

        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, title, (w // 2 - 200, h // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 2)
        cv2.putText(frame, str(secs_left), (w // 2 - 20, h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 100), 4)
        if subtitle:
            cv2.putText(frame, subtitle, (w // 2 - 150, h // 2 + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("HASTAVEDA — Mudra Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN COLLECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  HASTAVEDA  —  Mudra Dataset Collector")
    print("=" * 65)
    print(f"  Target     : {TOTAL_SAMPLES_PER_MUDRA} images per mudra")
    print(f"  Mudras     : {', '.join(MUDRAS)}")
    print(f"  Output dir : {DATASET_PATH}")
    print("=" * 65)
    print("  Controls : [Q] Skip mudra  [ESC] Save & quit\n")

    # Load any previously collected data for resume support
    data = load_existing_data()
    # Ensure all mudra keys exist
    for mudra in MUDRAS:
        if mudra not in data:
            data[mudra] = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check that it is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    interrupted = False

    for mudra in MUDRAS:
        # --- Resume: count how many we already have ---
        already_captured = count_existing_images(mudra)
        # Sync in-memory samples length to disk image count
        if len(data[mudra]) > already_captured:
            data[mudra] = data[mudra][:already_captured]
        elif already_captured > len(data[mudra]):
            already_captured = len(data[mudra])

        if already_captured >= TOTAL_SAMPLES_PER_MUDRA:
            print(f"  [{mudra}] Already complete ({already_captured} images). Skipping.")
            continue

        count = already_captured
        remaining = TOTAL_SAMPLES_PER_MUDRA - count
        print(f"\n[{mudra}] Resuming from {count}/{TOTAL_SAMPLES_PER_MUDRA} — {remaining} to go")

        # --- Preparation countdown ---
        ok = countdown(cap,
                       f"Prepare: {mudra.upper()}",
                       POSE_PREP_TIME,
                       f"Resuming from {count}")
        if not ok:
            interrupted = True
            break

        last_capture_time = time.time()
        skip_mudra = False

        while count < TOTAL_SAMPLES_PER_MUDRA:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame from webcam.")
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)

            current_time = time.time()
            captured_now  = False

            if results.multi_hand_landmarks:
                # Only process the FIRST detected hand
                hand_lm = results.multi_hand_landmarks[0]

                # Draw skeleton on the correct hand
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                landmarks_42, bbox = extract_landmarks(hand_lm, w, h)
                draw_bounding_box(frame, bbox)

                if current_time - last_capture_time >= CAPTURE_DELAY:
                    if len(landmarks_42) == 42:       # Safety check
                        data[mudra].append(landmarks_42)
                        save_annotated_image(frame, mudra, count)
                        count += 1
                        last_capture_time = current_time
                        captured_now = True

            extra = "✅ Capturing..." if captured_now else "⏳ No hand detected"
            draw_overlay(frame, mudra, count, TOTAL_SAMPLES_PER_MUDRA, extra)
            cv2.imshow("HASTAVEDA — Mudra Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print(f"  Skipped. Saved {count} images for [{mudra}].")
                skip_mudra = True
                break
            elif key == 27:   # ESC
                print(f"\n  Saving and quitting at [{mudra}] count={count}.")
                save_data(data)
                interrupted = True
                break

            # --- Mini break every 100 images ---
            if count > 0 and count % SAMPLES_PER_BREAK == 0 and count != TOTAL_SAMPLES_PER_MUDRA:
                print(f"  [{mudra}] {count} captured. Taking {BREAK_TIME_SMALL}s break...")
                save_data(data)   # Save progress during break
                ok = countdown(cap, "REST BREAK", BREAK_TIME_SMALL,
                               f"Resume in...  ({count}/{TOTAL_SAMPLES_PER_MUDRA})")
                if not ok:
                    interrupted = True
                    break

        if interrupted:
            break

        if not skip_mudra:
            save_data(data)
            print(f"  ✅ [{mudra}] Complete! {count} images collected.")

            if MUDRAS.index(mudra) < len(MUDRAS) - 1:
                print(f"  Taking {BREAK_TIME_LARGE}s break before next mudra...")
                ok = countdown(cap, "MUDRA COMPLETE!", BREAK_TIME_LARGE,
                               "Next mudra starting soon...")
                if not ok:
                    interrupted = True
                    break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Always save before exiting
    save_data(data)
    save_classes()

    print("\n" + "=" * 65)
    print("  Collection Summary:")
    print("=" * 65)
    total_images = 0
    for mudra in MUDRAS:
        n = count_existing_images(mudra)
        total_images += n
        status = "✅ Complete" if n >= TOTAL_SAMPLES_PER_MUDRA else f"⚠️  {n}/{TOTAL_SAMPLES_PER_MUDRA}"
        print(f"  {mudra:15s}  {status}")

    print(f"\n  Total images on disk : {total_images}")
    print(f"  mudra_samples.json   : saved")
    print(f"  mudra_classes.json   : saved")

    if interrupted:
        print("\n  ⚠️  Collection was interrupted. Re-run collect.py to continue.")
    else:
        print("\n  🎉 All mudras collected! Run train.py next.")
    print("=" * 65)


if __name__ == "__main__":
    main()
