# player_position.py

# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
#-------------------------------------------------------------------------------------------------#
# Import the video to break down frame by frame.
cap = cv2.VideoCapture('Play_Ex/Play_Ex_4.mp4')

#-------------------------------------------------------------------------------------------------#
# Setting thresholds.

# Define the primary colour thresholds for offensive team in example.
low_colour_off = np.array([120,60,70], dtype="uint8")
high_colour_off = np.array([138,128,118], dtype="uint8")

# Define the primary colour thresholds for defensive team in the example.
low_colour_def = np.array([75,95,220], dtype="uint8")
high_colour_def = np.array([170,170,255], dtype="uint8")

# Define the secondary colour thresholds for both teams. Is actually the same for off/def due to sharpness of the white for secondary colour.
low_colour_sec = np.array([245,245,245], dtype="uint8")
high_colour_sec = np.array([255,255,255], dtype="uint8")

# Thresholds to create a mask of NFL logo for removal.
low_colour_nfl_logo = np.array([90,90,50], dtype="uint8")
high_colour_nfl_logo = np.array([150,140,80], dtype="uint8")

# Find thresholds for luminance which corresponds to uniform.
lab_low = np.array([11,105,85])
lab_high = np.array([75,187,175])

#-------------------------------------------------------------------------------------------------#
# Storing the images to variables.
#while True:
    #-------------------------------------------------------------------------------------------------#
    
# Return current frame.
#ret_val, image = cap.read()

# Find the screenshot with the play footage.
image = cv2.imread('Play_Ex/Play Capture 4.PNG')

# Make a copy to allow for multiple outputs.
new_image = image.copy()

# Make another copy
img = image.copy()

# Convert the image to a L*a*b* colour space to fill out the sum mask further.
lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

#-------------------------------------------------------------------------------------------------#
# Creating the various masks from the colour thresholds.

# Create a mask with a single uniform colour for each team.
mask_img_off = cv2.inRange(image, low_colour_off, high_colour_off)
mask_img_def = cv2.inRange(image, low_colour_def, high_colour_def)

# Create a mask of the secondary colour thresholds.
mask_img_sec = cv2.inRange(image, low_colour_sec, high_colour_sec)

# Create the mask of the lab thresholds.
mask_lab = cv2.inRange(lab, lab_low, lab_high)

# Create a mask of the NFL logo.
mask_img_nfl_logo = cv2.inRange(image, low_colour_nfl_logo, high_colour_nfl_logo)

# Join the masks together excluding the NFL logo.
mask_img = mask_img_off + mask_img_def + mask_lab + mask_img_sec



#-------------------------------------------------------------------------------------------------#
# Create the contour arrays from the masks.

# Find contours of the unfiltered mask. Effectively 'Control' of filtering performance.
contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#-------------------------------------------------------------------------------------------------#
# Run filtering on the masks.

# Use adaptive thresholding to "binarize" the image.
thresh = cv2.adaptiveThreshold(mask_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 1)

thresh_nfl_logo = cv2.adaptiveThreshold(mask_img_nfl_logo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 1)

thresh_sec = cv2.adaptiveThreshold(mask_img_sec, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 1)

thresh_off = cv2.adaptiveThreshold(mask_img_off, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 1)

thresh_def = cv2.adaptiveThreshold(mask_img_def, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 1)

#-------------------------------------------------------------------------------------------------#
# Perform morphological filtering. 

# Perform some morphological operations to help distinguish some of the features in the image.
kernel = np.ones((2,2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Morph the secondary mask.
opening_sec = cv2.morphologyEx(thresh_sec, cv2.MORPH_OPEN, kernel, iterations=2)
closing_sec = cv2.morphologyEx(opening_sec, cv2.MORPH_CLOSE, kernel, iterations=2)

# Define the area where the logo is for removal based on contour.
kernel_nfl_logo = np.ones((7,7), np.uint8)
opening_nfl_logo = cv2.morphologyEx(thresh_nfl_logo, cv2.MORPH_OPEN, kernel_nfl_logo, iterations=6)
closing_nfl_logo = cv2.morphologyEx(opening_nfl_logo, cv2.MORPH_CLOSE, kernel_nfl_logo, iterations=2)

# Morph the primary offense.
opening_off = cv2.morphologyEx(thresh_off, cv2.MORPH_OPEN, kernel, iterations=2)
closing_off = cv2.morphologyEx(opening_off, cv2.MORPH_CLOSE, kernel, iterations=2)

# Morph the primary defense.
opening_def = cv2.morphologyEx(thresh_def, cv2.MORPH_OPEN, kernel, iterations=2)
closing_def = cv2.morphologyEx(opening_def, cv2.MORPH_CLOSE, kernel, iterations=2)

#-------------------------------------------------------------------------------------------------#
# Find the contours.

contours, hierarchy = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contours_nfl_logo, hierarchy_nfl_logo = cv2.findContours(closing_nfl_logo, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contours_sec, hierarchy_sec = cv2.findContours(closing_sec, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contours_off, hierarchy_off = cv2.findContours(closing_off, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contours_def, hierarchy_def = cv2.findContours(closing_def, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#-------------------------------------------------------------------------------------------------#

# Necessary, unsure.
try: hierarchy_nfl_logo = hierarchy_nfl_logo[0]

except: hierarchy_nfl_logo = []

# Define the variables.
height_nfl, width_nfl = mask_img_nfl_logo.shape
min_x_nfl, min_y_nfl = width_nfl, height_nfl
max_x_nfl = max_y_nfl = 0

# Computes contour coordinates.
for cont_nfl_logo, hier_nfl_logo in zip(contours_nfl_logo, hierarchy_nfl_logo):
    (x_nfl,y_nfl,w_nfl,h_nfl) = cv2.boundingRect(cont_nfl_logo)
    min_x_nfl, max_x_nfl = min(x_nfl, min_x_nfl), max(x_nfl+w_nfl, max_x_nfl)
    min_y_nfl, max_y_nfl = min(y_nfl, min_y_nfl), max(y_nfl+h_nfl, max_y_nfl)
    
    # Find the NFL logo rectangle.
    if x_nfl != 0 and y_nfl != 0:
        x0_nfl = x_nfl
        y0_nfl = y_nfl
        x1_nfl = x_nfl + w_nfl
        y1_nfl = y_nfl + h_nfl

#-------------------------------------------------------------------------------------------------#

# Necessary, unsure.
try: hierarchy = hierarchy[0]

except: hierarchy = []

# Define the variables.
height, width = mask_img.shape
min_x, min_y = width, height
max_x = max_y = 0

# computes the bounding box for the contour, and draws it on the frame,
for contour, hier in zip(contours, hierarchy):
    (x,y,w,h) = cv2.boundingRect(contour)   
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    # Find the field logo rectangle. Want to ignore all contours within the confines of the team logo rectangle.
    if w > 300 and h > 150 and w < 1000 and h < 750:
        x0 = x
        y0 = y
        x1 = x + w
        y1 = y + h

min_area = 20000 # Removes exterior contours.

# Reiterate through the list now that we have our team logo.
for contour, hier in zip(contours, hierarchy):  
    (x,y,w,h) = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if area > 10000:
        print("{} {} {} {}".format(x,y,w,h))
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    if area > min_area:
        pass # Dont draw the contours.
    elif (x >= x0_nfl) and (x <= x1_nfl) and (y >= y0_nfl) and (y <= y1_nfl):
        pass # Do nothing
    elif (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1):
        pass # Do nothing
    else:
        cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)   
        #plt.scatter(height - y, width - x)  

#-------------------------------------------------------------------------------------------------#

# Necessary, unsure.
try: hierarchy_sec = hierarchy_sec[0]

except: hierarchy_sec = []

# Define the variables.
height_sec, width_sec = mask_img_sec.shape
min_x_sec, min_y_sec = width_sec, height_sec
max_x_sec = max_y_sec = 0

radius = 5 # Radius for fitting circles.
center_full_array_sec = []
center_plotted_sec_array = []

# Computes contour coordinates.
for cont_sec, hier_sec in zip(contours_sec, hierarchy_sec):
    (x_sec,y_sec,w_sec,h_sec) = cv2.boundingRect(cont_sec)

    # Find the center of the contour rectangle.
    center = (int(x_sec + (w_sec / 2)), int(y_sec + (h_sec / 2)))

    proximity_flag_sec = False
    for i in range(0, len(center_full_array_sec)):
        x_exist, y_exist = center_full_array_sec[i]
        x, y = center      
        
        if x > x_exist - 10 and x < x_exist + 10 and y > y_exist - 10  and y < y_exist + 10:
            proximity_flag_sec = True
        elif (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1):
            proximity_flag_sec = True # Not proximity but from the logo.

    if not proximity_flag_sec:
        cv2.circle(img, tuple(center), radius, (255, 0, 0), 2)
        center_plotted_sec_array.append(center)
        plt.scatter(height - y, width - x)  

    #else:
        #cv2.circle(img, tuple(center), radius, (0, 255, 255), 2)

    center_full_array_sec.append(center)

#-------------------------------------------------------------------------------------------------#

# Necessary, unsure.
try: hierarchy_off = hierarchy_off[0]

except: hierarchy_off = []

# Define the variables.
height_off, width_off = mask_img_off.shape
min_x_off, min_y_off = width_off, height_off
max_x_off = max_y_off = 0

center_full_array_off = []
center_plotted_off_array = []

# computes the bounding box for the contour, and draws it on the frame,
for contour_off, hier_off in zip(contours_off, hierarchy_off):
    (x_off,y_off,w_off,h_off) = cv2.boundingRect(contour_off)   
    # Find the center of the contour rectangle.
    center = (int(x_off + (w_off / 2)), int(y_off + (h_off / 2)))

    proximity_flag_off = False
    for i in range(0, len(center_full_array_off)):
        x_exist, y_exist = center_full_array_off[i]
        x, y = center      
        
        if x > x_exist - 10 and x < x_exist + 10 and y > y_exist - 10  and y < y_exist + 10:
            proximity_flag_off  = True
        elif (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1):
            proximity_flag_off  = True # Not proximity but from the logo.

    if not proximity_flag_off :
        cv2.circle(img, tuple(center), radius, (0, 0, 255), 2)
        center_plotted_off_array.append(center)
        plt.scatter(height - y, width - x)  

    #else:
        #cv2.circle(img, tuple(center), radius, (0, 255, 255), 2)

    center_full_array_off.append(center)

#-------------------------------------------------------------------------------------------------#

# Necessary, unsure.
try: hierarchy_def = hierarchy_def[0]

except: hierarchy_def = []

# Define the variables.
height_def, width_def = mask_img_def.shape
min_x_def, min_y_def = width_def, height_def
max_x_def = max_y_def = 0

center_full_array_def = []
center_plotted_def_array = []

# computes the bounding box for the contour, and draws it on the frame,
for contour_def, hier_def in zip(contours_def, hierarchy_def):
    (x_def,y_def,w_def,h_def) = cv2.boundingRect(contour_def)   
    # Find the center of the contour rectangle.
    center = (int(x_def + (w_def / 2)), int(y_def + (h_def / 2)))

    proximity_flag_def = False
    for i in range(0, len(center_full_array_def)):
        x_exist, y_exist = center_full_array_def[i]
        x, y = center      
        
        if x > x_exist - 10 and x < x_exist + 10 and y > y_exist - 10  and y < y_exist + 10:
            proximity_flag_def  = True
        elif (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1):
            proximity_flag_def  = True # Not proximity but from the logo.

    if not proximity_flag_def :
        cv2.circle(img, tuple(center), radius, (0, 255, 0), 2)
        center_plotted_def_array.append(center)
        plt.scatter(height - y, width - x)  

    #else:
        #cv2.circle(img, tuple(center), radius, (0, 255, 255), 2)

    center_full_array_def.append(center)

#-------------------------------------------------------------------------------------------------#
# Detect the offensive players.
# For each red dot: if a blue within a certain 'minima' place a central point n white on the plot
center_plotted_sec_array.sort()
center_plotted_off_array.sort()
center_plotted_def_array.sort()

for i, j in center_plotted_sec_array:
    print("Sec {} {}".format(i, j))
for i, j in center_plotted_off_array:
    print("Off {} {}".format(i, j))
for i, j in center_plotted_def_array:
    print("Def {} {}".format(i, j))
#-------------------------------------------------------------------------------------------------#

# Draw the contours from the original mask for comparison.
new_image = cv2.drawContours(new_image, contours, -1, (0,255,0), 3, 8)
#img = cv2.drawContours(img, contours_sec, -1, (0,255,0), 3)

#-------------------------------------------------------------------------------------------------#

# Show the output.
while True:
    # Show the final output to the window. update output depending on filter used.

    # Show the colour space version of the image.
    cv2.imshow('L*a*b', lab)
    # Show the mask of the luminance values.
    cv2.imshow('Mask L*a*b', mask_lab)
    # Show the original image with rectangles around players.
    cv2.imshow('Original + Players Rectangled.', image)
    # Show the original image with contours around players.
    cv2.imshow('Original + Players Contoured', new_image)
    # Show the centroid of the secondary defined players.
    cv2.imshow('Original + Player secondary centroids', img)
    # Show the mask of the offence with the primary threshold.
    cv2.imshow('Mask offence - primary', mask_img_off)
    # Show the mask of the defence with the primary threshold.
    cv2.imshow('Mask defence - primary', mask_img_def)
    # Show the mask of the secondary thresholds.
    cv2.imshow('Mask defence - secondary', mask_img_sec)
    # Show the sum of the masks.
    cv2.imshow('Sum of all masks', mask_img)
    # Show the thresholded NFL logo.
    cv2.imshow('Thresholded NFL logo', thresh_nfl_logo)
    # Show the opened NFL logo.
    cv2.imshow('Opening NFL logo', opening_nfl_logo)
    # Show the closed NFL logo.
    cv2.imshow('Closed NFL logo', closing_nfl_logo)
    # Show the thresholded image.
    cv2.imshow('Thresholded', thresh)
    # Show the opened image.
    cv2.imshow('Opened', opening)
    # Show the closed image.
    cv2.imshow('Closed', closing)
    # Show the thresholded secondary image.
    cv2.imshow('Thresholded Secondary', thresh_sec)
    # Show the opened secondary image.
    cv2.imshow('Opened Secondary', opening_sec)
    # Show the closed secondary image.
    cv2.imshow('Closed Secondary', closing_sec)
    # Show the final plot.
    plt.show()

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


