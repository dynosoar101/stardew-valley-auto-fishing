#this script is purely for debugging
#uncomment necessary blocks to test certain aspects

#import libraries
import time
import pyautogui
from PIL import ImageGrab
import cv2
import numpy
import mss
import os

ASSETS = os.path.join(os.path.dirname(__file__), 'image_assets')
print(f"Looking in: {ASSETS}")
print("Files found:")
for f in os.listdir(ASSETS):
    print(f"  {f}")

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
'''
import mss
import cv2
import numpy

monitor_width = 2560
monitor_height = 1440

region = {
    
    "left":   int(monitor_width * 0.384),
    "top":    int(monitor_height * 0.209),
    "width":  int(monitor_width * 0.455) - int(monitor_width * 0.384),
    "height": int(monitor_height * 0.646) - int(monitor_height * 0.209)
}

with mss.mss() as sct:
    input("Press Enter when minigame is active...")
    frame = numpy.array(sct.grab(region))
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    
    cv2.imwrite('scan_region.png', img_bgr)
    print(f"Size: {w}x{h}")
    
    # Scan across ALL columns at mid-height to find where green/fish are
    mid_y = h // 2
    print(f"\nScanning horizontally at y={mid_y}:")
    for x in range(0, w, 5):
        px = hsv[mid_y, x]
        bgr = img_bgr[mid_y, x]
        print(f"  x={x}: HSV{tuple(int(v) for v in px)} BGR{tuple(int(v) for v in bgr)}")
    
    # Also find all green pixels and report where they are
    lower_green = numpy.array([40, 150, 150])
    upper_green = numpy.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    coords = numpy.where(mask > 0)
    if coords[0].size > 0:
        print(f"\nGreen pixels found!")
        print(f"  X range: {coords[1].min()} to {coords[1].max()}")
        print(f"  Y range: {coords[0].min()} to {coords[0].max()}")
    else:
        print("\nNo green pixels found with current range")
        '''