#import libraries
import time
import pyautogui
from PIL import ImageGrab
import cv2
import numpy

# Bobber detection - for future use
# Red range: #5a4444 (90, 68, 68) to #911303 (145, 19, 3)
# White range: #a1d4ce (161, 212, 206) to #fff5e2 (255, 245, 226)
# Strategy:
#   1. Scan screen for red pixels in range
#   2. For each red pixel, check 20x20 box around it for white pixels
#   3. If both found close together = bobber location
#   4. Use bobber location to find player position

#use appropriate monitor dimensions, mine are from 2560x1440. get yours via running debug.py script
monitor_width = 2560
monitor_height = 1440

def fish():
    while True:
        center_x = monitor_width // 2   # 1280
        center_y = monitor_height // 2  # 720
        screenshot = ImageGrab.grab(
            bbox=(center_x - 260, center_y - 390, center_x + 260, center_y + 390),
            all_screens=True
        )
        # Convert to OpenCV format
        img = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
        # Define the color range for yellow exclamation points 255, 232, 53
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # These ranges target the specific vibrant yellow/gold of the icon
        # H: 20-30 covers the yellow spectrum
        # S: 100-255 ensures we get the saturation of the gold
        # V: 200-255 targets the bright highlights of the exclamation mark
        lower_yellow = numpy.array([20, 100, 200])
        upper_yellow = numpy.array([30, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Create a mask for the yellow color
       # mask = cv2.inRange(img, lower_yellow, upper_yellow)
        # Check if any yellow pixels are detected
        if cv2.countNonZero(mask) > 0:
            print("Found yellow exclamation point")
            break 
        #sleep for 0.1 seconds
    return
        