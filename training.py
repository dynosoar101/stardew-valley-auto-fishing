import time
import os
import csv
import threading
import glob
import mss
import cv2
import numpy
import pandas as pd
from pynput import mouse, keyboard

# ── Output directory ──────────────────────────────────────────────────────────
SESSION = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.path.dirname(__file__), f"training_data/{SESSION}")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ── Capture region ────────────────────────────────────────────────────────────
REGION = {
    "left":   int(2560 * 0.420),
    "top":    int(1440 * 0.250),
    "width":  int(2560 * 0.455) - int(2560 * 0.420),
    "height": int(1440 * 0.66)  - int(1440 * 0.250),
}

# ── Shared input state ────────────────────────────────────────────────────────
mouse_held = False
state_lock = threading.Lock()

def on_click(x, y, button, pressed):
    global mouse_held
    if button == mouse.Button.left:
        with state_lock:
            mouse_held = pressed

def on_release(key):
    if key == keyboard.Key.f8:
        print("\nF8 pressed — stopping capture.")
        return False

# ── CSV log ───────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "labels.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame_file", "timestamp", "mouse_left"])

# ── Listeners ─────────────────────────────────────────────────────────────────
mouse_listener    = mouse.Listener(on_click=on_click)
keyboard_listener = keyboard.Listener(on_release=on_release)
mouse_listener.start()
keyboard_listener.start()

# ── 1. RECORD ─────────────────────────────────────────────────────────────────
print(f"Saving to: {OUT_DIR}")
print("Recording — press F8 to stop.\n")

frame_index = 0
try:
    with mss.MSS() as sct:
        while keyboard_listener.is_alive():
            t0 = time.perf_counter()

            frame = numpy.array(sct.grab(REGION))
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            with state_lock:
                held = int(mouse_held)

            fname = f"{frame_index:06d}.png"
            cv2.imwrite(os.path.join(FRAMES_DIR, fname), img_bgr)
            csv_writer.writerow([fname, f"{time.time():.6f}", held])

            frame_index += 1

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, (1/60) - elapsed))

except KeyboardInterrupt:
    print("\nCtrl+C — stopping.")

finally:
    mouse_listener.stop()
    keyboard_listener.stop()
    csv_file.flush()
    csv_file.close()
    print(f"\nDone. {frame_index} frames saved to {OUT_DIR}")

# ── 2. FILTER (only unprocessed sessions) ────────────────────────────────────
print("\nFiltering garbage frames...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
template = cv2.imread(os.path.join(SCRIPT_DIR, 'image_assets', 'template.png'), cv2.IMREAD_GRAYSCALE)

if template is None:
    raise FileNotFoundError("Could not load image_assets/template.png")

def is_valid_frame(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape != template.shape:
        resized = cv2.resize(template, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        resized = template
    result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
    _, conf, _, _ = cv2.minMaxLoc(result)
    return conf >= 0.60

sessions = glob.glob("training_data/*/labels.csv")
total_kept = 0
total_removed = 0

for session_csv in sessions:
    session_dir = os.path.dirname(session_csv)
    filtered_csv_path = os.path.join(session_dir, "labels_filtered.csv")

    # Skip already-processed sessions
    if os.path.exists(filtered_csv_path):
        continue

    frames_dir = os.path.join(session_dir, "frames")
    print(f"Processing: {session_dir}")

    with open(session_csv, "r") as f_in, \
         open(filtered_csv_path, "w", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["frame_file", "timestamp", "mouse_left"])

        kept = 0
        removed = 0

        for row in reader:
            fpath = os.path.join(frames_dir, row["frame_file"])

            if not os.path.exists(fpath):
                removed += 1
                continue

            img = cv2.imread(fpath)
            if img is None:
                os.remove(fpath)
                removed += 1
                continue

            if is_valid_frame(img):
                writer.writerow([row["frame_file"], row["timestamp"], row["mouse_left"]])
                kept += 1
            else:
                os.remove(fpath)
                removed += 1

        print(f"  Kept: {kept}  |  Removed: {removed}")
        total_kept    += kept
        total_removed += removed

    # Delete original now that filtered version exists
    os.remove(session_csv)

if total_kept == 0 and total_removed == 0:
    print("No new sessions to process.")
else:
    print(f"Total kept: {total_kept} | Total removed: {total_removed}")

# ── 3. PRINT CLASS BALANCE ────────────────────────────────────────────────────
print("\nClass balance across all sessions:")
dfs = [pd.read_csv(f) for f in glob.glob("training_data/*/labels_filtered.csv")]
combined = pd.concat(dfs)
print(combined["mouse_left"].value_counts())