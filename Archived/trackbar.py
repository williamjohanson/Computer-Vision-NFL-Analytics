# trackbar.py

# Imports
import cv2
import numpy as np

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

def __init__():
    """ Create trackbars for threshold values. """
    cv2.createTrackbar('CannyThreshold1', 'Play Example 1', 0, 1200, nothing)
    cv2.createTrackbar('CannyThreshold2', 'Play Example 1', 0, 1200, nothing)
    cv2.createTrackbar("HoughThreshold", 'Play Example 1', 0, 200, nothing)

def update_val():
    """ Return the trackbar position for threshold values. """
    houghThreshold = 260 #cv2.getTrackbarPos('HoughThreshold', 'Play Example 1')
    cannyThreshold1 = 350 #cv2.getTrackbarPos('CannyThreshold1', 'Play Example 1')
    cannyThreshold2 = 350 #cv2.getTrackbarPos('CannyThreshold2', 'Play Example 1')
    
    return houghThreshold, cannyThreshold1, cannyThreshold2
