import time
import pyautogui
from PIL import ImageGrab
import cv2
import numpy

import cast
import fishing
import minigame
#delay til script actually runs
print("script started, autofishing beginning in 5 seconds")
time.sleep(5)
print("script started")
while True:
    cast.cast()
    time.sleep(2)
    fishing.fish()
    #click
    pyautogui.mouseDown()   # hold left click
    time.sleep(0.5)           # wait for full power
    pyautogui.mouseUp()
    #time.sleep(1) 
    #end of fish method means a fish has been detected
    minigame.minigame()
