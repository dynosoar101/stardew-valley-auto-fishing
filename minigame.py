import time
import pyautogui
import mss
import cv2
import numpy

def minigame():
    monitor_width = 2560
    monitor_height = 1440

    minigame_top    = int(monitor_height * 0.250)
    minigame_bottom = int(monitor_height * 0.650)
    minigame_left   = int(monitor_width * 0.420) #was 400
    minigame_right  = int(monitor_width * 0.455) #was 0.475

    region = {
        "left": minigame_left,
        "top": minigame_top,
        "width": minigame_right - minigame_left,
        "height": minigame_bottom - minigame_top
    }

    # Tightened fish range - pure teal only, cuts out false positives
    lower_fish  = numpy.array([82, 120, 120])
    upper_fish  = numpy.array([95, 255, 255])

    lower_green = numpy.array([35, 50, 150])   # lowered saturation to catch pale green
    upper_green = numpy.array([90, 255, 255])

    DEBUG = True
    debug_count = 0

    # PID values - tune these if still overshooting
    # error = fish_y - bar_center (positive means fish is below bar)
    # hold_threshold: if error > this, hold mouse (fish above bar, need to rise)
    # release_threshold: if error < this, release mouse (fish below bar, need to fall)
    hold_threshold    = -20   # fish is this many pixels ABOVE bar center
    release_threshold =  20   # fish is this many pixels BELOW bar center

    # Pulse timing - instead of holding forever, pulse the click
    # This prevents the bar from flying past the fish
    HOLD_TIME    = 0.05   # seconds to hold per pulse
    RELEASE_TIME = 0.03   # seconds to release per pulse

    last_action = None

    with mss.mss() as sct:
        while True:
            frame = numpy.array(sct.grab(region))
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            fish_mask = cv2.inRange(hsv, lower_fish, upper_fish)
            bar_mask  = cv2.inRange(hsv, lower_green, upper_green)

            fish_coords = numpy.where(fish_mask > 0)
            bar_coords  = numpy.where(bar_mask > 0)

            if DEBUG and debug_count % 30 == 0:
                debug_img = img_bgr.copy()
                debug_img[fish_mask > 0] = [0, 0, 255]
                debug_img[bar_mask > 0]  = [0, 255, 0]
                cv2.imwrite(f'debug_{debug_count}.png', debug_img)
            debug_count += 1

            if fish_coords[0].size > 50 and bar_coords[0].size > 100:
                fish_y     = int(numpy.median(fish_coords[0]))
                bar_top    = int(numpy.min(bar_coords[0]))
                bar_bottom = int(numpy.max(bar_coords[0]))
                bar_center = (bar_top + bar_bottom) // 2

                error = fish_y - bar_center
                print(f"Fish Y: {fish_y}, Bar center: {bar_center}, Error: {error}")

                if error < hold_threshold:
                    # Fish is ABOVE bar - hold to rise toward fish
                    if last_action != 'hold':
                        pyautogui.mouseDown()
                        last_action = 'hold'
                    time.sleep(HOLD_TIME)

                elif error > release_threshold:
                    # Fish is BELOW bar - release to fall toward fish
                    if last_action != 'release':
                        pyautogui.mouseUp()
                        last_action = 'release'
                    time.sleep(RELEASE_TIME)

                # else: fish in dead zone, do nothing, keep last state

            else:
                # Don't change state on brief detection loss
                print(f"Not detected - fish:{fish_coords[0].size} bar:{bar_coords[0].size}")
                return