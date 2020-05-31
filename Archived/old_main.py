# main.py

# Imports.
import numpy as np
import cv2
import pytesseract as tess

# Define tess path.
tess.pytesseract.tesseract_cmd = r'C:\Users\willi\AppData\Local\tesseract.exe'

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

# Find the video file with the play footage.
cap = cv2.VideoCapture('Play_Ex/Play_Ex_1.mp4')

mirror = False

# Create a window for the output.
cv2.namedWindow('Play Example 1', cv2.WINDOW_AUTOSIZE)

# Set the Canny thresholds.
#cannyThreshold1 = 300
#cannyThreshold2 = 300
cv2.createTrackbar('CannyThreshold1', 'Play Example 1', 0, 1200, nothing)
cv2.createTrackbar('CannyThreshold2', 'Play Example 1', 0, 1200, nothing)
cv2.createTrackbar("HoughThreshold", 'Play Example 1', 0, 200, nothing)

# Set the Hough threshold.
#houghThreshold = 100

# Order is blue, green, red - CHECK
low_colour_bound = np.array([0,0,0], dtype="uint8")
high_colour_bound = np.array([255,255,255], dtype="uint8")

# Iterate through the video frames.
while True:
    # Return the trackbar position for threshold values.
    houghThreshold = cv2.getTrackbarPos('HoughThreshold', 'Play Example 1')
    cannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'Play Example 1')
    cannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'Play Example 1')

    ret_val, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(frame, low_colour_bound, high_colour_bound)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    frame = frame & mask_rgb

    if mirror:
        frame = cv2.flip(frame, 1)
        
    # Create a new copy of the frame for drawing on later
    img = frame.copy()

    # Find the text in the image if applicable.
    text = tess.image_to_string(img)
    print(text)

    # Find the Canny Edges.
    edges = cv2.Canny(frame, cannyThreshold1, cannyThreshold2)

    # Find straight lines.
    lines = cv2.HoughLines(edges, 1, np.pi/180, houghThreshold)

    # For each line that was detected, draw it on the img.
    if lines is not None:
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    # Create a combined video of Hough Line Transform result and the Canny edge detector.
    combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

    # Show the final output to the window.
    cv2.imshow('Play Example 1', combined)

    # Esc to quit.
    if cv2.waitKey(1) == 27:
        break 








# Everything is done, release the video capture object.
cap.release()

# Close all the frames.
cv2.destroyAllWindows()