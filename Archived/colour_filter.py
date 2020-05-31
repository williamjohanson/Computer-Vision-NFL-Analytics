# colour_filter.py

# Imports
import numpy as np
import cv2

def gray_filter(frame, low_colour_bound, high_colour_bound):
    """ Set the frame gray and then back to reduce noise? """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(frame, low_colour_bound, high_colour_bound)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return gray, mask_rgb

def init_colour_thresholds():
    """ Define high and low colour bounds for simple thresholding. """
    # Order is blue, green, red - CHECK
    low_colour_bound = np.array([220,220,220], dtype="uint8")
    high_colour_bound = np.array([255,255,255], dtype="uint8")
    
    return low_colour_bound, high_colour_bound

def uniform_colour_thresholds():
    low_colour_steelers = np.array([39,202,229], dtype="uint8")
    high_colour_steelers = np.array([160,240,250], dtype="uint8")
    low_colour_pats = np.array([43,56,62], dtype="uint8")
    high_colour_pats = np.array([62,79,93], dtype="uint8")

    return low_colour_steelers, high_colour_steelers, low_colour_pats, high_colour_pats