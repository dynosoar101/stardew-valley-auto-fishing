#this script is purely for debugging
#uncomment necessary blocks to test certain aspects

#import libraries
import time
import pyautogui
from PIL import ImageGrab
import cv2
import numpy


print("compiled fine")

#replace the appropiate block's screenshot imageGrab bbox with the monitor you want to capture. mine is a primary and secondary 2560x1440
#uncomment this to get all your specific monitor dimensions
'''
from screeninfo import get_monitors
for m in get_monitors():
    print(m)
'''
#primary monitor
#screenshot = ImageGrab.grab(bbox=(0, 0, 2560, 1440), all_screens=True)

#secondary monitor
#screenshot = ImageGrab.grab(bbox=(-1920, 0, 0, 1080), all_screens=True)



#this loop will scan a 30 x 30 pixel area around the cursor for the color pink and press windows key when found
'''
while True:
    #get mouse position
    pyautogui.position()
    x, y = pyautogui.position()
    # bbox should be (left, top, right, bottom)
    # so offset by 30 pixels in each direction for a 60x60 area
    bbox = (x - 30, y - 30, x + 30, y + 30)
    screenshot = ImageGrab.grab(bbox=bbox, all_screens=True)

    # Convert to OpenCV format
    img = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
    # Define the color range for pink
    lower_pink = numpy.array([140, 100, 100])
    upper_pink = numpy.array([160, 255, 255])

    # Create a mask for the pink color
    mask = cv2.inRange(img, lower_pink, upper_pink)
    # Check if any pink pixels are detected
    if cv2.countNonZero(mask) > 0:
        print("Pink detected!")
        pyautogui.press('win')
        break
    #sleep for 0.1 seconds
    time.sleep(0.1)
    print("end of loop")
'''

#this loop will scan the entire screen for the color pink and press windows key when found
'''
while True:
    # bbox should be (left, top, right, bottom)
    screenshot = ImageGrab.grab(bbox=(0, 0, 2560, 1440), all_screens=True)

    # Convert to OpenCV format
    img = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
    # Define the color range for pink
    lower_pink = numpy.array([140, 100, 100])
    upper_pink = numpy.array([160, 255, 255])

    # Create a mask for the pink color
    mask = cv2.inRange(img, lower_pink, upper_pink)
    # Check if any pink pixels are detected
    if cv2.countNonZero(mask) > 0:
        print("Pink detected!")
        pyautogui.press('win')
        break
    #sleep for 0.1 seconds
    time.sleep(0.1)
    print("end of loop")
    '''