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
    print("Error: fish.png does not have a transparent background.")
    exit()

fish_bgr  = fish_img_raw[:, :, :3]
fish_mask = fish_img_raw[:, :, 3]
fish_h, fish_w = fish_bgr.shape[:2]

FISHING_LEVEL = 10

def get_bar_height(level):
    # Wiki: base 96px + 8px per level, out of 568px total track
    # Our capture region is 590px tall — scale accordingly
    base_px = 96 + level * 8          # 176px at level 10 in wiki units
    scale   = 590 / 568               # our region vs wiki track height
    return int(base_px * scale)       # ~183px at level 10 in our coordinates

def is_valid_frame(gray_frame, threshold=0.6):
    result = cv2.matchTemplate(gray_frame, template_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= threshold

def find_fish_template(bgr_frame, threshold=0.85):
    result = cv2.matchTemplate(bgr_frame, fish_bgr, cv2.TM_CCORR_NORMED, mask=fish_mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        return max_loc[0], max_loc[1], max_val
    return None

def find_green_bar(bgr_frame, fish_y=None, level=0):
    """
    Finds the green catch bar and reconstructs its full position even when
    the fish is occluding part of it.

    Returns dict:
        top, bottom, center   — reconstructed bar bounds
        detected_top          — actual top of visible green
        detected_bottom       — actual bottom of visible green
        partial               — True if fish was cutting through bar
        source                — 'full' | 'partial_above' | 'partial_below'
    """
    expected_h = get_bar_height(level)
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 150, 150])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find all green blobs
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels < 2:
        return None

    # Collect all blobs with area > 50 (ignore tiny noise)
    blobs = [
        {
            "top":    stats[i, cv2.CC_STAT_TOP],
            "bottom": stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT],
            "height": stats[i, cv2.CC_STAT_HEIGHT],
            "area":   stats[i, cv2.CC_STAT_AREA],
        }
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] > 50
    ]

    if not blobs:
        return None

    # Sort by area descending — largest blob is the bar (or biggest chunk of it)
    blobs.sort(key=lambda b: b["area"], reverse=True)
    main = blobs[0]

    # Case 1: Full bar detected (height close to expected)
    if main["height"] >= expected_h * 0.80:
        return {
            "top":              main["top"],
            "bottom":           main["bottom"],
            "center":           (main["top"] + main["bottom"]) // 2,
            "detected_top":     main["top"],
            "detected_bottom":  main["bottom"],
            "partial":          False,
            "source":           "full",
        }

    # Case 2: Partial detection — fish is cutting through the bar
    # Use fish_y to figure out which side of the bar we're seeing
    if fish_y is not None:
        fish_center_y = fish_y + fish_h // 2

        if fish_center_y < main["top"]:
            # Fish is ABOVE the visible green chunk — we're seeing the bottom of the bar
            # Reconstruct: bar top = detected top - expected height + detected height
            reconstructed_top    = main["top"] - (expected_h - main["height"])
            reconstructed_bottom = main["bottom"]
            source = "partial_below_fish"  # fish above, bar below
        else:
            # Fish is BELOW or inside the visible chunk — we're seeing the top of the bar
            reconstructed_top    = main["top"]
            reconstructed_bottom = main["top"] + expected_h
            source = "partial_above_fish"  # fish below, bar above

        return {
            "top":              reconstructed_top,
            "bottom":           reconstructed_bottom,
            "center":           (reconstructed_top + reconstructed_bottom) // 2,
            "detected_top":     main["top"],
            "detected_bottom":  main["bottom"],
            "partial":          True,
            "source":           source,
        }

    # Case 3: Partial but no fish info — return what we have
    return {
        "top":              main["top"],
        "bottom":           main["bottom"],
        "center":           (main["top"] + main["bottom"]) // 2,
        "detected_top":     main["top"],
        "detected_bottom":  main["bottom"],
        "partial":          True,
        "source":           "partial_no_fish",
    }

# ── Main capture loop ─────────────────────────────────────────────────────────
stop_script = False
last_bar = None  # last known bar position

def on_press(key):
    global stop_script
    if key == keyboard.Key.esc:
        stop_script = True
        return False

listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Starting debug capture. Press ESC to stop...")
frame_index = 0

with mss.mss() as sct:
    while not stop_script:
        grabbed    = sct.grab(REGION)
        frame_bgr  = cv2.cvtColor(np.array(grabbed), cv2.COLOR_BGRA2BGR)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if is_valid_frame(frame_gray):

            # Fish detection
            fish_match = find_fish_template(frame_bgr)
            fish_y = None
            if fish_match is not None:
                fx, fy, fscore = fish_match
                fish_y = fy
                cv2.rectangle(frame_bgr, (fx, fy), (fx + fish_w, fy + fish_h), (0, 0, 255), 2)
                cv2.putText(frame_bgr, f"Fish {fscore:.2f}", (fx, max(0, fy - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Green bar detection
            bar = find_green_bar(frame_bgr, fish_y=fish_y, level=FISHING_LEVEL)

            if bar is not None:
                last_bar = bar
                color = (0, 255, 0) if not bar["partial"] else (0, 200, 255)  # green=full, orange=partial
                label = f"Bar [{bar['source']}] h={bar['bottom']-bar['top']}"
            elif last_bar is not None:
                bar = last_bar
                color = (0, 100, 255)  # red-orange = using last known
                label = "Bar [LAST KNOWN]"
            else:
                bar = None
                label = ""

            if bar is not None:
                frame_w = frame_bgr.shape[1]
                bx = 0
                bw = frame_w
                cv2.rectangle(frame_bgr, (bx, bar["top"]), (bx + bw, bar["bottom"]), color, 2)
                cv2.line(frame_bgr, (bx, bar["center"]), (bx + bw, bar["center"]), (0, 255, 255), 1)
                cv2.putText(frame_bgr, label, (2, max(0, bar["top"] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_index:04d}.png")
            cv2.imwrite(frame_path, frame_bgr)

            if frame_index % 10 == 0:
                fish_str = f"fish=y{fish_y}" if fish_match else "fish=None"
                bar_str  = f"bar={bar['top']}-{bar['bottom']} [{bar['source']}]" if bar else "bar=None"
                print(f"[{frame_index:04d}] {fish_str}  {bar_str}")

            frame_index += 1

        time.sleep(0.05)

print(f"\nDone. {frame_index} frames saved to: {FRAMES_DIR}")