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
        # gets a screenshot of a 100x100 pixel area at the center of the screen
        center_x = monitor_width // 2
        center_y = monitor_height // 2
        screenshot = ImageGrab.grab(bbox=(center_x - 200, center_y - 300, center_x + 200, center_y + 200), all_screens=True)

        # Convert to OpenCV format
        img = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
        # Define the color range for yellow exclamation points 255, 232, 53
        lower_yellow = numpy.array([0, 210, 230])
        upper_yellow = numpy.array([60, 245, 255])
        # Create a mask for the yellow color
        mask = cv2.inRange(img, lower_yellow, upper_yellow)
        # Check if any yellow pixels are detected
        if cv2.countNonZero(mask) > 0:
            break 
        #sleep for 0.1 seconds
        time.sleep(0.1)
        print("end of loop")
        