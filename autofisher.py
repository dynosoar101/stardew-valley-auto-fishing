import time
import pyautogui
from PIL import ImageGrab
import cv2
import numpy

import cast
import fishing
#delay til script actually runs
time.sleep(5)
while True:
    cast.cast()
    fishing.fish()
    time.sleep(1)