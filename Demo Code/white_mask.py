# draw_player_contours.py

# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Define the secondary colour thresholds for defensive team in the example.
low_colour_def_sec = np.array([250,250,250], dtype="uint8")
high_colour_def_sec = np.array([255,255,255], dtype="uint8")

# Find the screenshot with the play footage.
image = cv2.imread('Play_Ex/Play Capture 4.PNG')

# Create the mask
mask_img = cv2.inRange(image, low_colour_def_sec, high_colour_def_sec)

while True:
    cv2.imshow('Img', image)
    cv2.imshow('Mask img', mask_img)

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break