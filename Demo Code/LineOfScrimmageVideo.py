# LineOfScrimmage.py

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time

# Code Setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Reporting/output.avi', fourcc, 20.0, (1280,720))

# Find the video file with the play footage.
cap = cv2.VideoCapture('Play_Ex/Play_Ex_6.mp4')

# Define the primary colour thresholds for defensive team in the example.
low_colour_def = np.array([75,95,220], dtype="uint8")
high_colour_def = np.array([170,170,255], dtype="uint8")

# Define the primary colour thresholds for offensive team in example.
low_colour_off = np.array([120,60,70], dtype="uint8")
high_colour_off = np.array([138,128,118], dtype="uint8")

img_height = 778
img_width = 1381

while True: 
    # Return current frame.
    ret_val, img_original = cap.read()

    #img_original = cv2.imread('Play_Ex/Play Capture 4.PNG')
    blur = cv2.GaussianBlur(img_original, (9,9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    minLineLength = 10
    maxLineGap = 1000
    cannyThreshold1 = 20
    cannyThreshold2 = 60

    # Create a new copy of the original image for drawing on later.
    img = img_original.copy()
    # Use the Canny Edge Detector to find some edges.
    edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
    # Attempt to detect straight lines in the edge detected image.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

    cv2.imshow('Blur', blur)
    cv2.imshow('Gray', gray)
    cv2.imshow('Edges', edges)

    grad_array = []
    lines_len = 0

    drawn_line_list = []

    # For each line that was detected, draw it on the img.
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:

                # Clear out all the border lines.
                if (x1 == x2) or (y1 == y2):
                    # Do nothing since these are effectively noise.
                    pass
                    
                elif abs((y2 - y1)) > abs((x2 - x1)):
                    new_line_flag = True
                    for start_pos in drawn_line_list:
                        if ((x1 > start_pos) and x1 < (start_pos + 100)) or (x1 > (start_pos - 100) and (x1 < start_pos)):
                            print(line)
                            new_line_flag = False

                    if new_line_flag:    
                        cv2.line(img_original,(x1,y1),(x2,y2),(0,255,255),5)

                        mean = int((x1 + x2) / 2)

                        plt.plot([0, img_height], [img_width - x1, img_width - x1], linewidth=0.5, color='black')
                        grad = (y2 - y1) / (x2 - x1)
                        print("Grad: {:.2f} ".format(grad))
                        if grad > 0:
                            grad_array.append(grad)
                            lines_len += 1
                        

                    drawn_line_list.append(x1)

                else:
                    # Do nothing since these are horizontal lines. Development -- find the hashmarkers. Research paper used L*a*b?
                    pass

    grad_array.sort()    

    if (len(grad_array) % 2) == 0:
        i = int(len(grad_array) / 2)
        grad_average = (grad_array[i] + grad_array[i-1]) / 2

    if (len(grad_array) % 2) != 0:
        i = int(len(grad_array) / 2)
        grad_average = (grad_array[i + 1] + grad_array[i] + grad_array[i-1]) / 3


    print("Grad Average: {:.2f}".format(grad_average))

    # Create a mask with a single uniform colour for each team.
    mask_img_def = cv2.inRange(img_original, low_colour_def, high_colour_def)
    mask_img_off = cv2.inRange(img_original, low_colour_off, high_colour_off)

    # Need to filter small spots off our mask.
    kernel = np.ones((2,2), np.uint8)
    mask_img_def = cv2.morphologyEx(mask_img_def, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_img_off = cv2.morphologyEx(mask_img_off, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find the contours from both masks.
    contours_def, hierarchy_def = cv2.findContours(mask_img_def, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_off, hierarchy_off = cv2.findContours(mask_img_off, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Unsure, but is necessary for zipping function?
    try: hierarchy_def = hierarchy_def[0]

    except: hierarchy_def = []

    height_def, width_def = mask_img_def.shape
    min_x_def, min_y_def = width_def, height_def
    max_x_def = max_y_def = 0

    x_2_def = 0
    y_2_def = 0

    # Computes the bounding box for the contour, and draws it on the frame,
    for contour_def, hier_def in zip(contours_def, hierarchy_def):
        (x_def,y_def,w_def,h_def) = cv2.boundingRect(contour_def)
        min_x_def, max_x_def = min(x_def, min_x_def), max(x_def+w_def, max_x_def)
        min_y_def, max_y_def = min(y_def, min_y_def), max(y_def+h_def, max_y_def)

        print("Min x {} min y {} max x {} max y {}".format(min_x_def, min_y_def, max_x_def, max_y_def))

        if max_x_def > x_2_def:
            x_2_def = max_x_def
            y_2_def = max_y_def
        
        #cv2.rectangle(img_original, (x_def,y_def), (x_def+w_def,y_def+h_def), (0, 255, 0), 2)

        #plt.scatter(x_def,y_def) 

    try: hierarchy_off = hierarchy_off[0]

    except: hierarchy_off = []

    # Maybe swap min and max and keep min as the origin (0,0)
    height_off, width_off = mask_img_off.shape
    min_x_off, min_y_off = width_off, height_off
    max_x_off = max_y_off = 0

    x_2_off = 1381
    y_2_off = 0

    # Computes the bounding box for the contour, and draws it on the frame,
    for contour_off, hier_off in zip(contours_off, hierarchy_off):
        (x_off,y_off,w_off,h_off) = cv2.boundingRect(contour_off)
        min_x_off, max_x_off = min(x_off, min_x_off), max(x_off+w_off, max_x_off)
        min_y_off, max_y_off = min(y_off, min_y_off), max(y_off+h_off, max_y_off)

        if (min_x_off < x_2_off) and (min_x_off > x_2_def):
            x_2_off = min_x_off
            y_2_off = min_y_off
        #cv2.rectangle(img_original, (x_off,y_off), (x_off+w_off,y_off+h_off), (255, 0, 0), 2)

        #plt.scatter(x_off,y_off) 



    x_1_off = int(x_2_off - y_2_off/grad_average)
    x_3_off = int((img_height-y_2_off)/grad_average + x_2_off)

    x_1_def = int(x_2_def - y_2_def/grad_average)
    x_3_def = int((img_height-y_2_def)/grad_average + x_2_def)

    x_1 = int((x_1_off + x_1_def) / 2)
    x_3 = int((x_3_off + x_3_def) / 2)

    #cv2.line(img_original,(x2_max,0),(x2_max,778),(255,255,255),10)
    #cv2.line(img_original,(x_1_off,0),(x_3_off,778),(255,255,255),10)
    #cv2.line(img_original,(x_1_def,0),(x_3_def,778),(255,255,255),10)
    cv2.line(img_original,(x_1,0),(x_3,img_height),(255,255,255),10)

    mean_los = int((x_1 + x_3) / 2)
    plt.plot([0, img_height], [img_width - mean_los, img_width - mean_los], linewidth=3, color='yellow')

    # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
    combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)

#while True:

    cv2.imshow('Hough Line Transform', combined)
    cv2.imshow('Contours', img_original)
    cv2.imshow('Mask Img Off', mask_img_off)
    cv2.imshow('Mask Img Def', mask_img_def)

    out.write(img_original)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

plt.show()
