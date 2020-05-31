import pytesseract as tess
from PIL import Image
import cv2

# Define tess path.
tess.pytesseract.tesseract_cmd = r'C:\Users\willi\AppData\Local\tesseract.exe'

# Allocate image file to variable.
img = cv2.imread("Play_Ex/Play Capture 3.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the text!
text = tess.image_to_string(img)

print(text)