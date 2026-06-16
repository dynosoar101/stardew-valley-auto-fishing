import time
import os
import csv
import threading
import mss
import cv2
import numpy
from pynput import mouse, keyboard

# ── Output directory ──────────────────────────────────────────────────────────
SESSION = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.path.dirname(__file__), f"training_data/{SESSION}")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ── Capture region (unchanged from minigame.py) ───────────────────────────────
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

# ── Capture loop ──────────────────────────────────────────────────────────────
print(f"Saving to: {OUT_DIR}")
print("Recording — press F8 to stop.\n")

frame_index = 0
try:
    with mss.mss() as sct:
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