#this script is simply to cast the rod. the time to cast is fixed at 1 second.

#import libraries
import time
import pyautogui

def cast():
    pyautogui.mouseDown()   # hold left click
    time.sleep(0.93)        # wait for full power
    pyautogui.mouseUp()   # release left click
    time.sleep(2)     #half second delay before termination