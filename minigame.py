import time
import os
import csv
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

LOG_PATH = os.path.join(OUT_DIR, "physics_log.csv")

# ── Config ────────────────────────────────────────────────────────────────────

FISHING_LEVEL = 10

REGION = {
    "left":   int(2560 * 0.420),
    "top":    int(1440 * 0.250),
    "width":  int(2560 * 0.470) - int(2560 * 0.435),
    "height": int(1440 * 0.660) - int(1440 * 0.250),
}

# ── Physics model ─────────────────────────────────────────────────────────────
# Starting guesses only — the model recalibrates GRAVITY and CLICK_ACCEL live
# from observed detection deltas during clean press/release windows.
GRAVITY_INIT     = 1250.0   # px/s² downward, refined at runtime
CLICK_ACCEL_INIT = 2100.0   # px/s² upward while held, refined at runtime

# Blend rate for online recalibration of gravity/click-accel from real samples.
ACCEL_LEARN_RATE = 0.20

# When a fresh detection arrives, snap hard toward it (detection is reliable
# per testing) rather than slow-blending — this is what "recalibrate the
# instant the bar is found again" means in practice.
DETECTION_SNAP_ALPHA = 0.85

# Bar detection
BAR_HSV_LOWER = np.array([43,  68, 196])
BAR_HSV_UPPER = np.array([78, 255, 229])
BAR_MIN_AREA  = 50

# Gap-merge: the fish sprite is drawn ON TOP of the green bar and its
# non-green pixels (blue/white/red scales) punch a hole through the color
# mask, splitting one continuous bar into 2+ fragments. Fragments separated
# by a small vertical gap are treated as one bar occluded by the fish,
# rather than picking only the single largest fragment (which previously
# produced wrong "partial" reconstructions whenever the fish overlapped
# the bar — confirmed by comparing fish bbox rows directly against the
# color-mask gap rows in captured frames).
BAR_MERGE_GAP       = 70    # px — generous for the fish sprite's height
BAR_MERGED_MIN_AREA = 300   # discard merged blobs smaller than this as noise

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
    """
    Detect the green catch-bar, merging fragments that were split apart by
    the fish sprite occluding part of the bar. Fragments separated by a
    small vertical gap (<= BAR_MERGE_GAP) are combined into one bar before
    height/partial logic runs, so the fish passing over the bar no longer
    produces a falsely-short "partial" reading.
    """
    expected_h = get_bar_height(level)
    hsv  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BAR_HSV_LOWER, BAR_HSV_UPPER)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels < 2:
        return None

    frags = sorted(
        [
            (stats[i, cv2.CC_STAT_TOP],
             stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT],
             stats[i, cv2.CC_STAT_AREA])
            for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] > BAR_MIN_AREA
        ],
        key=lambda f: f[0]
    )
    if not frags:
        return None

    # ── Gap-merge fragments split by an occluder (the fish sprite) ────────
    merged = []
    cur_top, cur_bot, cur_area = frags[0]
    for top, bot, area in frags[1:]:
        if top - cur_bot <= BAR_MERGE_GAP:
            cur_bot   = max(cur_bot, bot)
            cur_area += area
        else:
            merged.append((cur_top, cur_bot, cur_area))
            cur_top, cur_bot, cur_area = top, bot, area
    merged.append((cur_top, cur_bot, cur_area))

    # Discard merged blobs too small to plausibly be the real bar
    merged = [m for m in merged if m[2] >= BAR_MERGED_MIN_AREA]
    if not merged:
        return None

    main_top, main_bot, main_area = max(merged, key=lambda m: m[2])
    main_height = main_bot - main_top

    if main_height >= expected_h * 0.80:
        top, bottom, partial, source = main_top, main_bot, False, "full"
    elif fish_y is not None:
        fish_center_y = fish_y + fish_h // 2
        if fish_center_y < main_top:
            top    = main_top - (expected_h - main_height)
            bottom = main_bot
            source = "partial_fish_above"
        else:
            top    = main_top
            bottom = main_top + expected_h
            source = "partial_fish_below"
        partial = True
    else:
        top, bottom, partial, source = main_top, main_bot, True, "partial_no_fish"

    return {"top": top, "bottom": bottom, "center": (top + bottom) // 2,
            "partial": partial, "source": source}


# ── Physics model with online recalibration ──────────────────────────────────

class BarPhysics:
    """
    Tracks the bar's center via a gravity + click-impulse model, and refines
    its own GRAVITY / CLICK_ACCEL estimates from real detection data whenever
    a clean single-state (press-only or release-only) window is observed
    between two fresh, trustworthy detections.

    Calibration logic:
      - We log (timestamp, detected_center) every time a FRESH detection
        succeeds, plus every press/release edge from the input listeners.
      - When two consecutive fresh detections happen with NO press/release
        edge in between (pure constant-state interval), the observed
        acceleration during that interval is a clean physics sample:
            observed_accel = 2 * (delta_x - v0*dt) / dt^2
        which we don't have v0 for directly, so instead we use the
        finite-difference of *velocity* across three consecutive fresh
        samples (so we don't need to know v0 a priori):
            v_mid = (x2 - x0) / (t2 - t0)   [central difference]
        and accel = (v_late - v_early) / dt, blended into our running
        estimate for whichever state (pressing/released) was constant
        across the window.
    """
    def __init__(self):
        self.center   = None
        self.vel      = 0.0
        self.last_t   = None
        self.pressing = False

        # Live-calibrated constants (start from init guesses)
        self.gravity     = GRAVITY_INIT
        self.click_accel = CLICK_ACCEL_INIT

        # Rolling history of fresh detections: (t, center, pressing_state)
        # Used to derive real accel samples for recalibration.
        self._fresh_history = []
        self._HISTORY_MAX = 5

    # ── Detection-driven update ────────────────────────────────────────────
    def seed(self, detected_center):
        self.center = float(detected_center)
        self.vel    = 0.0
        self.last_t = time.perf_counter()
        self._fresh_history.clear()
        self._fresh_history.append((self.last_t, self.center, self.pressing))

    def correct(self, detected_center):
        """Called every time a FRESH (non-stale) detection succeeds."""
        now = time.perf_counter()

        if self.center is None:
            self.seed(detected_center)
            return

        dt = now - self.last_t if self.last_t else 0.016
        delta = detected_center - self.center

        # Hard-ish snap toward the trustworthy fresh detection.
        implied_vel = delta / dt if dt > 0 else 0.0
        self.center = (1 - DETECTION_SNAP_ALPHA) * self.center + DETECTION_SNAP_ALPHA * detected_center
        self.vel    = (1 - DETECTION_SNAP_ALPHA) * self.vel    + DETECTION_SNAP_ALPHA * implied_vel
        self.last_t = now

        # ── Record this fresh sample and attempt recalibration ────────────
        self._fresh_history.append((now, float(detected_center), self.pressing))
        if len(self._fresh_history) > self._HISTORY_MAX:
            self._fresh_history.pop(0)
        self._try_recalibrate()

    def _try_recalibrate(self):
        """
        Look for a clean 3-sample window in fresh-detection history where the
        press state did NOT change, and derive an empirical acceleration
        from the central-difference velocity change. Blend into our running
        gravity / click_accel estimate.
        """
        h = self._fresh_history
        if len(h) < 3:
            return

        t0, x0, p0 = h[-3]
        t1, x1, p1 = h[-2]
        t2, x2, p2 = h[-1]

        # Need constant press-state across the whole window to attribute
        # the observed acceleration to a single physical regime.
        if not (p0 == p1 == p2):
            return

        dt_01 = t1 - t0
        dt_12 = t2 - t1
        if dt_01 <= 0 or dt_12 <= 0:
            return

        v_early = (x1 - x0) / dt_01
        v_late  = (x2 - x1) / dt_12
        dt_mid  = ((t2 - t0) / 2.0)
        if dt_mid <= 0:
            return

        observed_accel = (v_late - v_early) / dt_mid

        # Sanity bounds — reject wild outliers (occlusion glitches, etc.)
        if abs(observed_accel) > 6000 or abs(observed_accel) < 50:
            return

        if p1:  # was pressing the whole window → net accel = gravity - click_accel
            implied_click_accel = self.gravity - observed_accel
            implied_click_accel = float(np.clip(implied_click_accel, 200, 8000))
            self.click_accel = ((1 - ACCEL_LEARN_RATE) * self.click_accel
                                + ACCEL_LEARN_RATE * implied_click_accel)
        else:   # was released the whole window → net accel = gravity
            implied_gravity = float(np.clip(observed_accel, 100, 5000))
            self.gravity = ((1 - ACCEL_LEARN_RATE) * self.gravity
                            + ACCEL_LEARN_RATE * implied_gravity)

    # ── Integration step (runs every frame regardless of detection) ───────
    def step(self):
        if self.center is None:
            self.last_t = time.perf_counter()
            return None

        now = time.perf_counter()
        dt  = now - self.last_t
        self.last_t = now

        accel = self.gravity
        if self.pressing:
            accel -= self.click_accel

        self.vel    += accel * dt
        self.center += self.vel * dt

        region_h = REGION["height"]
        self.center = float(np.clip(self.center, 0, region_h))

        return self.center

    def on_press(self):
        self.pressing = True
        self._fresh_history.clear()   # state changed — don't mix regimes

    def on_release(self):
        self.pressing = False
        self._fresh_history.clear()   # state changed — don't mix regimes


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
    # Stardew Valley also accepts Spacebar, C, and X for the fishing minigame
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
        (f"g={physics.gravity:.0f} a={physics.click_accel:.0f}", (200, 200, 0)),
    ]
    for i, (ln, col) in enumerate(hud_lines):
        cv2.putText(vis, ln, (2, 12 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, ln, (2, 12 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col,     1, cv2.LINE_AA)

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
print(f"Physics log: {LOG_PATH}")

log_file = open(LOG_PATH, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "frame_idx", "t_perf", "fish_y", "fish_score",
    "bar_top", "bar_bottom", "bar_center", "bar_partial", "bar_source",
    "phys_center", "phys_vel", "gravity", "click_accel",
    "pressing", "is_stale",
])

frame_idx = 0

try:
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

            if bar_detected is not None and not bar_detected.get("partial"):
                physics.correct(bar_detected["center"])

            bar_estimated = physics.step()

            log_writer.writerow([
                frame_idx,
                f"{time.perf_counter():.6f}",
                fish_y if fish_y is not None else "",
                f"{fish_match[2]:.4f}" if fish_match else "",
                bar_detected["top"]    if bar_detected else "",
                bar_detected["bottom"] if bar_detected else "",
                bar_detected["center"] if bar_detected else "",
                bar_detected.get("partial") if bar_detected else "",
                bar_detected.get("source")  if bar_detected else "",
                f"{bar_estimated:.2f}" if bar_estimated is not None else "",
                f"{physics.vel:.2f}",
                f"{physics.gravity:.1f}",
                f"{physics.click_accel:.1f}",
                int(physics.pressing),
                int(is_stale),
            ])

            if frame_idx % 10 == 0:
                fish_str = f"fish=y{fish_y}" if fish_match else "fish=None"
                if bar_detected:
                    bar_str = f"det={bar_detected['top']}-{bar_detected['bottom']} [{bar_detected['source']}]"
                else:
                    bar_str = "det=None"
                phys_str = f"phys_c={bar_estimated:.1f}" if bar_estimated else "phys=uninit"
                print(f"[{frame_idx:04d}] {fish_str}  {bar_str}  {phys_str}  "
                      f"vel={physics.vel:+.1f}  g={physics.gravity:.0f} a={physics.click_accel:.0f}  "
                      f"{'CLICK' if physics.pressing else 'idle'}")

            vis = draw_debug(frame_bgr, fish_match, bar_detected, bar_estimated,
                             is_stale, physics.pressing, frame_idx)

            disp = cv2.resize(vis, (vis.shape[1] * 3, vis.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Debug Capture", disp)

            cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{frame_idx:04d}.png"), vis)

            frame_idx += 1

            if frame_idx % 50 == 0:
                log_file.flush()

            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.016)

finally:
    cv2.destroyAllWindows()
    mouse_listener.stop()
    key_listener.stop()
    log_file.flush()
    log_file.close()
    print(f"\nDone. {frame_idx} frames saved to: {FRAMES_DIR}")
    print(f"Physics log saved to: {LOG_PATH}")