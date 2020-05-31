# extract_text.py

# Imports.
import numpy as np
import cv2
import pytesseract as tess

# Define tess path.
tess.pytesseract.tesseract_cmd = r'C:\Users\willi\AppData\Local\tesseract.exe'


def tess_func(gray):
    """ Function to utilise tess text reading for each frame. """
    h = gray.shape[0]
    print(h)

    # Find the image height  
    boxes = tess.image_to_boxes(gray)

    # Create a original copy of the frame.
    img = gray.copy()

    # Draw the bounding boxes on the image.
    for line in boxes.splitlines():
        data = line.split(' ')
        gray = cv2.rectangle(gray,
                        (int(data[1]), h - int(data[2])), (int(data[3]), h - int(data[4])),
                        (0, 0, 255), 1)


    # Find the text in the image if applicable.
    text = tess.image_to_string(img)
    print("Tess text: " + text)

    return gray                               
    