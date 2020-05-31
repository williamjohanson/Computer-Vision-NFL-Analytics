# LOS code is more functional right now for field lines. Need to allow for plotting of lines onto plot in 90 degree orientation.
# LineOfScrimmage.py

# Imports
import cv2
import numpy as np

# Find the video file with the play footage.
cap = cv2.VideoCapture('Play_Ex/Play_Ex_6.mp4')

while True: 
    # Return current frame.
    ret_val, img_original = cap.read()

    #img_original = cv2.imread('Play_Ex/Play Capture 3.PNG')
    blur = cv2.GaussianBlur(img_original, (9,9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Hough Line Transform')

    minLineLength = 10
    maxLineGap = 200
    cannyThreshold1 = 20
    cannyThreshold2 = 80

    # Create a new copy of the original image for drawing on later.
    img = img_original.copy()
    # Use the Canny Edge Detector to find some edges.
    edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
    # Attempt to detect straight lines in the edge detected image.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

    print(lines)
    m = 0

    # Line No.
    line_no = 1

    # For each line that was detected, draw it on the img.
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:

                # Clear out all the border lines.
                if (x1 == x2) or (y1 == y2):
                    # Do nothing since these are effectively noise.
                    pass
                    
                elif abs((y2 - y1)) > abs((x2 - x1)):
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

                else:
                    # Do nothing since these are horizontal lines. Development -- find the hashmarkers. Research paper used L*a*b?
                    pass
                
                line_no += 1

    # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
    combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

    #while True:

    cv2.imshow('Hough Line Transform', combined)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

#####################################################################################################################################################


# field_lines.py

# Imports.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pytesseract as tess

# Define tess path.
tess.pytesseract.tesseract_cmd = r'C:\Users\willi\AppData\Local\tesseract.exe'

# Define key variables.
houghThreshold = 180
cannyThreshold1 = 350 
cannyThreshold2 = 350 

# Order is blue, green, red - BGR
low_colour_bound = np.array([125,125,125], dtype="uint8")
high_colour_bound = np.array([255,255,255], dtype="uint8")

# Find the screenshot with the play footage.
frame = cv2.imread('Play_Ex/Play Capture 4.PNG')
cap = frame.copy()

# Create a window for the output.
cv2.namedWindow('Play Example 1', cv2.WINDOW_AUTOSIZE)

# Colour filter.
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mask = cv2.inRange(frame, low_colour_bound, high_colour_bound)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

frame = frame & mask_rgb

# Find the Canny Edges.
edges = cv2.Canny(frame, cannyThreshold1, cannyThreshold2)

# Find straight lines.
lines = cv2.HoughLines(edges, 1, np.pi/180, houghThreshold)

print(lines)

# Draw them on the current frame/image.
values = []
if lines is not None:
    for line in lines:
        for rho, theta in line:
            if (theta > 2.3) and (len(lines) >=20): # Careful with theta depending on camera orientation, can vary.
                print("Rho is {}, theta is {}".format(rho, theta))
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)    
                values.append([a,b,x0,y0])   

# Would like to implement filtering to only draw one line for each spot.
for a,b,x0,y0 in values: 
    plt.scatter(x0,y0)  

h = cap.shape[0]
print(h)
boxes = tess.image_to_boxes(cap)

# Convert the text!
text = tess.image_to_string(cap)
print(text)

# Draw the bounding boxes on the image.
for line in boxes.splitlines():
    data = line.split(' ')
    cap = cv2.rectangle(cap,
                        (int(data[1]), h - int(data[2])), (int(data[3]), h - int(data[4])),
                        (0, 0, 255), 1)

while True:
    # Show the final output to the window. update output depending on filter used
    cv2.imshow('Play Example 1', frame)
    cv2.imshow('Outline field text', cap)
    plt.show()

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

lines.release()
lines.destroyAllWindows()