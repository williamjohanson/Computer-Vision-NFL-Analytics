# main.py

# Imports.
import numpy as np
import cv2
import pytesseract as tess

# Define tess path.
tess.pytesseract.tesseract_cmd = r'C:\Users\willi\AppData\Local\tesseract.exe'

# Find the video file with the play footage.
cap = cv2.imread('Play_Ex/Play Capture 4.png')

h = cap.shape[0]
print(h)
boxes = tess.image_to_boxes(cap)

# Draw the bounding boxes on the image.
for line in boxes.splitlines():
    data = line.split(' ')
    cap = cv2.rectangle(cap,
                        (int(data[1]), h - int(data[2])), (int(data[3]), h - int(data[4])),
                        (0, 0, 255), 1)

# Create a window for the output.
cv2.namedWindow('Play Example 1', cv2.WINDOW_AUTOSIZE)

# Iterate through the video frames.
while True:                                 
    # Show the final output to the window.
    cv2.imshow('Play Example 1', cap)

    # Esc to quit.
    if cv2.waitKey(1) == 27:
        break 





# Everything is done, release the video capture object.
cap.release()

# Close all the frames.
cv2.destroyAllWindows()