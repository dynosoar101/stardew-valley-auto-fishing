import time
import os
import mss
import cv2
import numpy as np
from pynput import keyboard

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "image_assets", "template.png")
FISH_PATH = os.path.join(SCRIPT_DIR, "image_assets", "fish.png")

SESSION = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(SCRIPT_DIR, f"debug_frames/{SESSION}")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

REGION = {
    "left":   int(2560 * 0.420), 
    "top":    int(1440 * 0.250),
    "width":  int(2560 * 0.470) - int(2560 * 0.435),
    "height": int(1440 * 0.660) - int(1440 * 0.250),
}

template_img = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if template_img is None:
    print(f"Error: Could not load template at {TEMPLATE_PATH}")
    exit()

fish_img_raw = cv2.imread(FISH_PATH, cv2.IMREAD_UNCHANGED)
if fish_img_raw is None:
    print(f"Error: Could not load fish at {FISH_PATH}")
    exit()

if fish_img_raw.shape[2] != 4:
    print("Error: fish.png does not have a transparent background (alpha channel).")
    exit()

# Extract the BGR color channels and the Alpha channel separately
fish_bgr = fish_img_raw[:, :, :3]
fish_mask = fish_img_raw[:, :, 3]
fish_h, fish_w = fish_bgr.shape[:2]

# ── Vision Functions ──────────────────────────────────────────────────────────

def is_valid_frame(gray_frame, threshold=0.6):
    """Returns True if the minigame UI is detected in the frame."""
    result = cv2.matchTemplate(gray_frame, template_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= threshold

def find_fish_template(bgr_frame, threshold=0.85):
    """
    Finds the fish using a masked template match to ignore the transparent background.
    """
    frame_width = bgr_frame.shape[1]
    left_half_bgr = bgr_frame[:, :int(frame_width * 0.5)]
    
    result = cv2.matchTemplate(bgr_frame, fish_bgr, cv2.TM_CCORR_NORMED, mask=fish_mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        return max_loc[0], max_loc[1], max_val
    return None


def find_green_bar(bgr_frame):
    """
    Finds the dynamic green catching bar using HSV masking.
    Uses a custom 3-Step Geometric Filter (Smear -> Shave -> Restore) 
    to handle heavy fading, fish occlusion, and seaweed noise simultaneously.
    """
    frame_width = bgr_frame.shape[1]
    
    # 1. Isolate the center track
    strip_left = int(frame_width * 0.25)
    strip_right = int(frame_width * 0.55)
    center_strip = bgr_frame[:, strip_left:strip_right]
    
    hsv_frame = cv2.cvtColor(center_strip, cv2.COLOR_BGR2HSV)
    
    # 2. Ultra-wide bounds to catch heavily faded bars
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    
    # --- 3. THE 3-STEP GEOMETRIC FILTER ---
    
    # STEP A: The Vertical Smear (cv2.dilate)
    # Smears pixels vertically by 45 pixels. 
    # This bridges the fish gap BEFORE we attempt to delete noise, 
    # ensuring the faint slivers of the bar aren't accidentally deleted.
    kernel_smear = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    mask_smeared = cv2.dilate(mask, kernel_smear)
    
    # STEP B: The Horizontal Shave (cv2.erode)
    # Erases any white pixels that aren't at least 15 pixels wide.
    # Seaweed is thin, so it completely vanishes. The smeared bar is wide, so it survives.
    kernel_shave = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    mask_shaved = cv2.erode(mask_smeared, kernel_shave)
    
    # STEP C: The Restore (cv2.erode)
    # Step A made the bar artificially taller. This shrinks it vertically 
    # back to its exact original top and bottom coordinates.
    mask_final = cv2.erode(mask_shaved, kernel_smear)
    
    # --- 4. Find Bounding Box ---
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # The true bar is the largest surviving contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Check against a minimum height to prevent random blips
        if h >= 15:
            real_x = x + strip_left
            return real_x, y, w, h
            
    return None

# ── Main Capture Loop ─────────────────────────────────────────────────────────
stop_script = False

def on_press(key):
    global stop_script
    if key == keyboard.Key.esc:
        stop_script = True
        return False

# Start keyboard listener for graceful exit
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Starting debug capture. Press ESC to stop...")
frame_index = 0

with mss.mss() as sct:
    while not stop_script:
        grabbed = sct.grab(REGION)
        frame_bgr = cv2.cvtColor(np.array(grabbed), cv2.COLOR_BGRA2BGR)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if is_valid_frame(frame_gray):
            
            # --- 1. Detect and Draw Fish ---
            fish_match = find_fish_template(frame_bgr)
            if fish_match is not None:
                fx, fy, fscore = fish_match
                cv2.rectangle(frame_bgr, (fx, fy), (fx + fish_w, fy + fish_h), (0, 0, 255), 2)
                cv2.putText(frame_bgr, f"Fish: {fscore:.2f}", (fx, max(0, fy - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # --- 2. Detect and Draw Green Bar ---
            bar_match = find_green_bar(frame_bgr)
            if bar_match is not None:
                bx, by, bw, bh = bar_match
                cv2.rectangle(frame_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(frame_bgr, "Bar", (bx, max(0, by - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Save the annotated frame
            frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_index:04d}.png")
            cv2.imwrite(frame_path, frame_bgr)
            
            if frame_index % 10 == 0:
                print(f"Captured {frame_index} valid frames so far...")
                
            frame_index += 1
            
        time.sleep(0.05)

print(f"\nDone. {frame_index} debug frames saved to: {FRAMES_DIR}")