# draw_lines.py

# Imports
import numpy as np
import cv2
import time

def draw(rho, theta):
    """ For each line that was detected, draw it on the img. """   
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    return x1, x2, y1, y2, x0, y0

def outline_players(mask, values, new_image):
    """ Place a circle around recognised player colours. """
    #min = 10000
    '''try:
        for _,_,_,_,_,y0 in values:
            if y0 < min:
                min = y0
    except: 
        min = 1000'''

    # Find a series of points which outline the shape in the mask.
    _, contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_image = cv2.drawContours(new_image, contours, -1, (0,255,0), -1, 8)
    #print("start {} finish".format(contours))

    return new_image


'''
    try:
        for contour in contours:
            # Fit a circle to the points of the contour.
            (x,y),radius = cv2.minEnclosingCircle(contour)
            if y < min:
                center = np.array([int(x),int(y)])
                radius = int(radius)
            i = 0
            if center is not None:
                print(i)
                i += 1
                # Draw circle around the ball.
                new_image = cv2.circle(new_image, tuple(center), radius,(0,255,0), 2)
                # Draw the center (not centroid!) of the ball.
                new_image = cv2.circle(new_image, tuple(center), 1,(0,255,0), 2)
    except:
        print("No objects to detect")
        '''

    