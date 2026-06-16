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
    minigame_left   = int(monitor_width * 0.420)
    minigame_right  = int(monitor_width * 0.455)

    region = {
        "left": minigame_left,
        "top": minigame_top,
        "width": minigame_right - minigame_left,
        "height": minigame_bottom - minigame_top
    }

    lower_fish  = numpy.array([83, 130, 130])
    upper_fish  = numpy.array([93, 255, 255])
    lower_green = numpy.array([35, 50, 150])
    upper_green = numpy.array([90, 255, 255])

    