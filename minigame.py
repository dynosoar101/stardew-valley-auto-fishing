import time
import os
import mss
import cv2
import numpy as np
from pynput import keyboard, mouse as pynmouse

# ── Paths & session ───────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "image_assets", "template.png")
FISH_PATH     = os.path.join(SCRIPT_DIR, "image_assets", "fish.png")

SESSION    = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR    = os.path.join(SCRIPT_DIR, f"debug_frames/{SESSION}")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

FISHING_LEVEL = 10

REGION = {
    "left":   int(2560 * 0.420),
    "top":    int(1440 * 0.250),
    "width":  int(2560 * 0.470) - int(2560 * 0.435),
    "height": int(1440 * 0.660) - int(1440 * 0.250),
}

# Physics model constants — calibrated to frame data
# GRAVITY:     px/s² the bar falls when mouse is up
# CLICK_ACCEL: px/s² upward impulse applied while mouse is held
GRAVITY     = 1250.0   # px/s² downward
CLICK_ACCEL = 2100.0   # px/s² upward while mouse held

# How aggressively to snap the physics model to a fresh detection.
# Lowered to 0.25 to trust the internal physics more and ignore partial-detection jitter.
CORRECTION_ALPHA = 0.25

# Bar detection
BAR_HSV_LOWER = np.array([43,  68, 196])
BAR_HSV_UPPER = np.array([78, 255, 229])
BAR_MIN_AREA  = 50

# ── Assets ────────────────────────────────────────────────────────────────────

template_img = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if template_img is None:
    raise FileNotFoundError(f"Could not load template: {TEMPLATE_PATH}")

fish_img_raw = cv2.imread(FISH_PATH, cv2.IMREAD_UNCHANGED)
if fish_img_raw is None:
    raise FileNotFoundError(f"Could not load fish: {FISH_PATH}")
if fish_img_raw.shape[2] != 4:
    raise ValueError("fish.png must have alpha channel")

fish_bgr  = fish_img_raw[:, :, :3]
fish_mask = fish_img_raw[:, :, 3]
fish_h, fish_w = fish_bgr.shape[:2]

# ── Detection helpers ─────────────────────────────────────────────────────────

def get_bar_height(level):
    base_px = 96 + level * 8
    scale   = 590 / 568
    return int(base_px * scale)

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
    expected_h = get_bar_height(level)
    hsv  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BAR_HSV_LOWER, BAR_HSV_UPPER)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels < 2:
        return None

    blobs = [
        {
            "top":    stats[i, cv2.CC_STAT_TOP],
            "bottom": stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT],
            "height": stats[i, cv2.CC_STAT_HEIGHT],
            "area":   stats[i, cv2.CC_STAT_AREA],
        }
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] > BAR_MIN_AREA
    ]
    if not blobs:
        return None

    main = max(blobs, key=lambda b: b["area"])

    if main["height"] >= expected_h * 0.80:
        top, bottom, partial, source = main["top"], main["bottom"], False, "full"
    elif fish_y is not None:
        fish_center_y = fish_y + fish_h // 2
        if fish_center_y < main["top"]:
            top    = main["top"] - (expected_h - main["height"])
            bottom = main["bottom"]
            source = "partial_fish_above"
        else:
            top    = main["top"]
            bottom = main["top"] + expected_h
            source = "partial_fish_below"
        partial = True
    else:
        top, bottom, partial, source = main["top"], main["bottom"], True, "partial_no_fish"

    return {"top": top, "bottom": bottom, "center": (top + bottom) // 2,
            "partial": partial, "source": source}


# ── Physics model ─────────────────────────────────────────────────────────────

class BarPhysics:
    def __init__(self):
        self.center   = None 
        self.vel      = 0.0  
        self.last_t   = None
        self.pressing = False 

    def seed(self, detected_center):
        self.center = float(detected_center)
        self.vel    = 0.0
        self.last_t = time.perf_counter()

    def correct(self, detected_center):
        if self.center is None:
            self.seed(detected_center)
            return
        delta = detected_center - self.center
        now = time.perf_counter()
        dt  = now - self.last_t if self.last_t else 0.016
        implied_vel = delta / dt if dt > 0 else 0.0
        self.center += CORRECTION_ALPHA * delta
        self.vel     = (1 - CORRECTION_ALPHA) * self.vel + CORRECTION_ALPHA * implied_vel
        self.last_t  = now

    def step(self):
        if self.center is None:
            self.last_t = time.perf_counter()
            return None

        now = time.perf_counter()
        dt  = now - self.last_t
        self.last_t = now

        accel = GRAVITY
        if self.pressing:
            accel -= CLICK_ACCEL 

        self.vel    += accel * dt
        self.center += self.vel * dt

        region_h = REGION["height"]
        self.center = float(np.clip(self.center, 0, region_h))

        return self.center

    def on_press(self):
        self.pressing = True

    def on_release(self):
        self.pressing = False


# ── Input listeners ───────────────────────────────────────────────────────────

physics     = BarPhysics()
stop_script = False

click_log   = []
MAX_LOG     = 200 

def _on_mouse_click(x, y, button, pressed):
    if button == pynmouse.Button.left:
        now = time.perf_counter()
        if pressed:
            physics.on_press()
            click_log.append((now, "down"))
        else:
            physics.on_release()
            click_log.append((now, "up"))
        if len(click_log) > MAX_LOG:
            click_log.pop(0)

def _on_key_press(key):
    global stop_script
    if key == keyboard.Key.esc:
        stop_script = True
        return False
    # Expanded listener: Stardew valley accepts Spacebar, C, and X 
    if key == keyboard.Key.space or (hasattr(key, 'char') and key.char in ('c', 'x', 'C', 'X')):
        physics.on_press()
        click_log.append((time.perf_counter(), "down"))
        if len(click_log) > MAX_LOG:
            click_log.pop(0)

def _on_key_release(key):
    if key == keyboard.Key.space or (hasattr(key, 'char') and key.char in ('c', 'x', 'C', 'X')):
        physics.on_release()
        click_log.append((time.perf_counter(), "up"))
        if len(click_log) > MAX_LOG:
            click_log.pop(0)

mouse_listener = pynmouse.Listener(on_click=_on_mouse_click)
key_listener   = keyboard.Listener(on_press=_on_key_press, on_release=_on_key_release)
mouse_listener.start()
key_listener.start()


# ── Debug overlay ─────────────────────────────────────────────────────────────

def draw_debug(frame_bgr, fish_match, bar_detected, bar_estimated,
               is_stale, pressing, frame_idx):
    vis = frame_bgr.copy()
    fw  = vis.shape[1]

    if bar_detected is not None:
        col   = (0, 255, 0) if not bar_detected.get("partial") else (0, 200, 255)
        label = f"Det [{bar_detected['source']}] h={bar_detected['bottom']-bar_detected['top']}"
        cv2.rectangle(vis, (0, bar_detected["top"]), (fw, bar_detected["bottom"]), col, 2)
        cv2.line(vis, (0, bar_detected["center"]), (fw, bar_detected["center"]),
                 (0, 255, 255), 1)
        cv2.putText(vis, label, (2, max(8, bar_detected["top"] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

    if bar_estimated is not None:
        est_cy = int(bar_estimated)
        dash_len, gap_len = 6, 4
        x = 0
        while x < fw:
            cv2.line(vis, (x, est_cy), (min(x + dash_len, fw), est_cy), (255, 0, 255), 1)
            x += dash_len + gap_len
        src = "PHYS" + (" [STALE]" if is_stale else "")
        cv2.putText(vis, src, (2, max(8, est_cy - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 0, 255), 1)

    if fish_match is not None:
        fx, fy, fscore = fish_match
        fish_cy = fy + fish_h // 2
        cv2.rectangle(vis, (fx, fy), (fx + fish_w, fy + fish_h), (0, 0, 255), 2)
        cv2.line(vis, (0, fish_cy), (fw, fish_cy), (0, 0, 180), 1)
        cv2.putText(vis, f"Fish {fscore:.2f}", (fx, max(0, fy - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)

    mouse_col  = (0, 255, 0) if pressing else (0, 0, 255)
    mouse_str  = "CLICK" if pressing else "IDLE"
    stale_str  = " [STALE]" if is_stale else ""
    hud_lines  = [
        (f"F:{frame_idx}{stale_str}", (255, 255, 255)),
        (f"Mouse:{mouse_str}",        mouse_col),
    ]
    for i, (ln, col) in enumerate(hud_lines):
        cv2.putText(vis, ln, (2, 12 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, ln, (2, 12 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col,     1, cv2.LINE_AA)

    fh       = vis.shape[0]
    strip_h  = 8
    strip_y  = fh - strip_h
    now      = time.perf_counter()
    window_s = 2.0
    cv2.rectangle(vis, (0, strip_y), (fw, fh), (30, 30, 30), -1)
    if click_log:
        for idx in range(len(click_log) - 1):
            t0, ev0 = click_log[idx]
            t1, _   = click_log[idx + 1]
            if t1 < now - window_s:
                continue
            age0 = now - t0
            age1 = now - t1
            x0 = int((1.0 - age0 / window_s) * fw)
            x1 = int((1.0 - age1 / window_s) * fw)
            if ev0 == "down":
                cv2.rectangle(vis, (x0, strip_y), (x1, fh), (0, 220, 0), -1)

    return vis


# ── Main loop ─────────────────────────────────────────────────────────────────

print("Debug capture running. Click freely — recording mouse + detection.")
print("Press ESC to stop.")

frame_idx = 0

with mss.mss() as sct:
    while not stop_script:
        grabbed    = sct.grab(REGION)
        frame_bgr  = cv2.cvtColor(np.array(grabbed), cv2.COLOR_BGRA2BGR)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if not is_valid_frame(frame_gray):
            print("Minigame UI not detected — waiting...")
            time.sleep(0.1)
            continue

        fish_match = find_fish_template(frame_bgr)
        fish_y     = fish_match[1] if fish_match else None

        bar_detected = find_green_bar(frame_bgr, fish_y=fish_y, level=FISHING_LEVEL)
        is_stale     = bar_detected is None

        if bar_detected is not None:
            if physics.center is None:
                physics.seed(bar_detected["center"])
            else:
                physics.correct(bar_detected["center"])

        bar_estimated = physics.step()  

        if frame_idx % 10 == 0:
            fish_str = f"fish=y{fish_y}" if fish_match else "fish=None"
            if bar_detected:
                bar_str = f"det={bar_detected['top']}-{bar_detected['bottom']} [{bar_detected['source']}]"
            else:
                bar_str = "det=None"
            phys_str = f"phys_c={bar_estimated:.1f}" if bar_estimated else "phys=uninit"
            print(f"[{frame_idx:04d}] {fish_str}  {bar_str}  {phys_str}  "
                  f"vel={physics.vel:+.1f}  {'CLICK' if physics.pressing else 'idle'}")

        vis = draw_debug(frame_bgr, fish_match, bar_detected, bar_estimated,
                         is_stale, physics.pressing, frame_idx)

        disp = cv2.resize(vis, (vis.shape[1] * 3, vis.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Debug Capture", disp)

        cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png"), vis)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.016)

cv2.destroyAllWindows()
mouse_listener.stop()
key_listener.stop()
print(f"\nDone. {frame_idx} frames saved to: {FRAMES_DIR}")