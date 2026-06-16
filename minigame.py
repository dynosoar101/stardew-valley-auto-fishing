import time
import pyautogui
from PIL import ImageGrab
import cv2
import numpy

def minigame():
    monitor_width = 2560
    monitor_height = 1440
    
    # Minigame panel scan region
    minigame_left   = int(monitor_width * 0.384)  # ~983
    minigame_top    = int(monitor_height * 0.209)  # ~301
    minigame_right  = int(monitor_width * 0.448)  # ~1147
    minigame_bottom = int(monitor_height * 0.646)  # ~930

   #fish icon range
    lower_fish = numpy.array([120, 120, 0])
    upper_fish = numpy.array([230, 160, 100])
        
    #green bar icon range
    lower_green = numpy.array([0, 240, 120])
    upper_green = numpy.array([50, 255, 200])


    while True:
        # gets a screenshot of a 100x100 pixel area at the center of the screen
        screenshot = ImageGrab.grab(bbox=(minigame_left, minigame_top, minigame_right, minigame_bottom), all_screens=True)

        img_bgr = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Create masks
        fish_mask = cv2.inRange(hsv, lower_fish, upper_fish)
        bar_mask = cv2.inRange(hsv, lower_green, upper_green)

        #check for fish
        fish_coords = numpy.where(fish_mask > 0)
        bar_coords = numpy.where(bar_mask > 0)

        if fish_coords[0].size >= 0:
            fish_y = (numpy.mean(fish_coords[0]))
            bar_center_y = (numpy.mean(bar_coords[0]))
            print(f"Fish Y: {fish_y}, Bar Y: {bar_center_y}")
            if fish_y < bar_center_y:
                pyautogui.mouseDown()
                print("Holding click - moving bar up")
            else:
                pyautogui.mouseUp()
                print("Releasing - letting bar fall")
        else:
            print(f"Not detected - fish pixels: {fish_coords[0].size}, bar pixels: {bar_coords[0].size}")
                
        