# draw_player_contours.py

# Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#################################### CODE SETUP ###################################################
# Define the primary colour thresholds for offensive team in example.
low_colour_off = np.array([120,60,70], dtype="uint8")
high_colour_off = np.array([138,128,118], dtype="uint8")

# Define the primary colour thresholds for defensive team in the example.
low_colour_def = np.array([75,95,220], dtype="uint8")
high_colour_def = np.array([170,170,255], dtype="uint8")
'''
# Define the secondary colour thresholds for offensive team in example.
low_colour_off_sec = np.array([0,200,195], dtype="uint8")
high_colour_off_sec= np.array([160,255,255], dtype="uint8")
'''
# Define the secondary colour thresholds for defensive team in the example. Is actually the same due to sharpness of the white for off secondary colour.
low_colour_def_sec = np.array([225,225,225], dtype="uint8")
high_colour_def_sec = np.array([255,255,255], dtype="uint8")

low_colour_logo = np.array([90,90,50], dtype="uint8")
high_colour_logo = np.array([150,140,80], dtype="uint8")

# Find the screenshot with the play footage.
image = cv2.imread('Play_Ex/Play Capture 4.PNG')
new_image = image.copy()

###################################################################################################
lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

l_channel,a_channel,b_channel = cv2.split(lab)

# Print the minimum and maximum of lightness.
print(np.min(l_channel)) # 11
print(np.max(l_channel)) # 255

# Print the minimum and maximum of a.
print(np.min(a_channel)) # 105
print(np.max(a_channel)) # 187

# Print the minimum and maximum of b.
print(np.min(b_channel)) # 85
print(np.max(b_channel)) # 175

lab_low = np.array([11,105,85])
lab_high = np.array([75,187,175])
mask_lab = cv2.inRange(lab, lab_low, lab_high)

cv2.imshow('L*a*b', lab)
cv2.imshow('Mask L*a*b', mask_lab)

mask_img_logo = cv2.inRange(image, low_colour_logo, high_colour_logo)

# Create a mask with a single uniform colour for each team.
mask_img_off = cv2.inRange(image, low_colour_off, high_colour_off)
mask_img_def = cv2.inRange(image, low_colour_def, high_colour_def)
#mask_img_off_sec = cv2.inRange(image, low_colour_off_sec, high_colour_off_sec)
mask_img_def_sec = cv2.inRange(image, low_colour_def_sec, high_colour_def_sec)

# Join the masks together.
mask_img = mask_img_off + mask_img_def + mask_lab + mask_img_def_sec


# Draw contours of the unfiltered mask
contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
new_image = cv2.drawContours(new_image, contours, -1, (0,255,0), 3, 8)

corners_frame = image.copy()
corners = cv2.cornerHarris(mask_img,2,3,0.04) # Tis very faint
# Dilate the detected corners to make them clearer in the output image.
corners = cv2.dilate(corners,None)
corners_frame[corners > 0.1 * corners.max()] = [0,0,255]
   
cv2.imshow("Harris", corners_frame)

# Segmentation of the players from features such as logo.
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
#mask_closed = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
#mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

# Use adaptive thresholding to "binarize" the image.
# MEAN_C is squarer pixelations than Gaussian. We shall go with Gaussian.
#thresh = cv2.adaptiveThreshold(mask_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)
thresh = cv2.adaptiveThreshold(mask_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
thresh_logo = cv2.adaptiveThreshold(mask_img_logo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)

# Perform some morphological operations to help distinguish some of the features in the image.
kernel = np.ones((2,2), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Define the area where the logo is for removal based on contour.
kernel_logo = np.ones((7,7), np.uint8)
opening_logo = cv2.morphologyEx(thresh_logo, cv2.MORPH_OPEN, kernel_logo, iterations=6)
closing_logo = cv2.morphologyEx(opening_logo, cv2.MORPH_CLOSE, kernel_logo, iterations=2)

# Now that we have hopefully distinguished the coins, find and fit ellipses around the coins in the image.
contours, hierarchy = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours_logo, hierarchy_logo = cv2.findContours(closing_logo, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

try: hierarchy = hierarchy[0]

except: hierarchy = []

height_l, width_l = mask_img_logo.shape
min_x_l, min_y_l = width_l, height_l
max_x_l = max_y_l = 0

# Computes contour coordinates.
for cont_logo, hier_logo in zip(contours_logo, hierarchy_logo):
    (x_l,y_l,w_l,h_l) = cv2.boundingRect(cont_logo)
    min_x_l, max_x_l = min(x_l, min_x_l), max(x_l+w_l, max_x_l)
    min_y_l, max_y_l = min(y_l, min_y_l), max(y_l+h_l, max_y_l)
    print('{} {} {} {}'.format(x_l,w_l,y_l,h_l))
    # Find the NFL logo rectangle.
    x0_l = x_l
    y0_l = y_l
    x1_l = x_l + w_l
    y1_l = y_l + h_l

height, width = mask_img.shape
min_x, min_y = width, height
max_x = max_y = 0

# computes the bounding box for the contour, and draws it on the frame,
for contour, hier in zip(contours, hierarchy):
    (x,y,w,h) = cv2.boundingRect(contour)
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    # Find the logo rectangle.
    '''if w > 150 and h > 150 and w < 1000 and h < 750:
        x0 = x
        y0 = y
        x1 = x + w
        y1 = y + h'''
        #print('{} {} {} {}'.format(x0,x1,y0,y1))
        #cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)
    # Ignore all contours within the confines of the team logo rectangle.

# Reiterate through the list now that we have our team logo??
#print('{} {} {} {}'.format(x0,x1,y0,y1))
for contour, hier in zip(contours, hierarchy):  
    (x,y,w,h) = cv2.boundingRect(contour)
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    #if (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1):
    #    k = "Do nothing" # Dont draw the contours.
    if (x >= x0_l) and (x <= x1_l) and (y >= y0_l) and (y <= y1_l):
        k = "Do nothing" # Dont draw the contours.
    else:
        cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)
        
        plt.scatter(height - y,width - x)  
        

#if max_x - min_x > 0 and max_y - min_y > 0:
#    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

#for contour in contours:
#    area = cv2.contourArea(contour)
#    print(area)
#    if area > 10000:  # Set a upper bound on the ellipse area, to remove field outline and logo (internal logo and NFL symbol how tho.)
#        continue

'''if len(contour) < 5:  # The fitEllipse function requires at least five points on the contour to function.
        continue'''

#    cv2.drawContours(image, contour, -1, (0,0,255), 5)
    #ellipse = cv2.fitEllipse(contour)  # Fit an ellipse to the points in the contour.
    #cv2.ellipse(image, ellipse, (0,255,0), 2)  # Draw the ellipse on the original image.

'''
# Blob detect and outline the blobs onto original image.
_, contours_off = cv2.findContours(mask_img_off, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
_, contours_def = cv2.findContours(mask_img_def, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


_, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 2000:  # Set a lower bound on the elipse area.
        continue

    if len(contour) < 5:  # The fitEllipse function requires at least five points on the contour to function.
        continue

ellipse = cv2.fitEllipse(contours)  # Fit an ellipse to the points in the contour.
cv2.ellipse(image, ellipse, (0,255,0), 2)  # Draw the ellipse on the original image.
'''
'''
print(contours_off)
print(contours_def)

cv2.drawContours(image, contours_off, -1, (0,255,0), 3)
cv2.drawContours(image, contours_def, -1, (0,0,255), 3)
# Further development into two uniform colours and then rectangle around player.

# Create a class of uniform colour thresholds to simply address based on teams playing and home/away


combined = np.concatenate((mask_img_off, mask_img_def), axis=1)
'''
#combined = np.concatenate((thresh, thresh2), axis=1)


# Show the output.
while True:
    # Show the final output to the window. update output depending on filter used
    cv2.imshow('Original', image)
    cv2.imshow('Original v2.0', new_image)
    cv2.imshow('Mask offence', mask_img_off)
    cv2.imshow('Mask defence', mask_img_def)
    cv2.imshow('Mask defence second', mask_img_def_sec)
    
    cv2.imshow('Colour masked', mask_img)
    cv2.imshow('Thresholded logo', thresh_logo)
    cv2.imshow('Opening logo', opening_logo)
    cv2.imshow('Closed logo', closing_logo)
    cv2.imshow('Thresholded', thresh)
    cv2.imshow('Opened', opening)
    cv2.imshow('Closed', closing)
    cv2.imshow('Corners', corners)
    plt.show()

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


