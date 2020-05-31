# main.py

# Imports.
import numpy as np
import cv2
import trackbar
import extract_text
import colour_filter
import draw_lines
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Find the video file with the play footage.
cap = cv2.VideoCapture('Play_Ex/Play_Ex_1.mp4')

# Unsure but seen defined as such elsewhere.
mirror = False

# Create a window for the output.
cv2.namedWindow('Play Example 1', cv2.WINDOW_AUTOSIZE)

# Initialise the trackbars.
#trackbar.__init__()

# Initialise the colour bounds.
low_colour_bound, high_colour_bound = colour_filter.init_colour_thresholds()

def filtering_main():
    """ Allow for various options of filtering by calling different filtering functions to act upon the frames/videos within the function. """
    # Create while loop to iterate through video frame by frame.
    while True:
        # Return trackbar positions.
        houghThreshold, cannyThreshold1, cannyThreshold2 = trackbar.update_val()

        # Return current frame.
        ret_val, frame = cap.read()

        # Create a copy of each frame for colour referencing.
        img = frame.copy()
        img2 = frame.copy()

        # Colour filter.
        gray, mask_rgb = colour_filter.gray_filter(frame, low_colour_bound, high_colour_bound)

        frame = frame & mask_rgb

        # Again, unsure, but seen it used.
        if mirror:
            frame = cv2.flip(frame, 1)
        
        # Initialise affine transform.
        #rows, cols, ch = frame.shape
        #pts1 = np.float32([[50,50],[200,50],[50,200]])
        #pts2 = np.float32([[10,100],[200,50],[100,250]])
        #M = cv2.getAffineTransform(pts1,pts2)
        #dst = cv2.warpAffine(frame, M, (cols, rows))
        #plt.subplot(121),plt.imshow(frame),plt.title('Input')
        #plt.subplot(121),plt.imshow(dst),plt.title('Input')
        #plt.show()

        # Find frame text and draw bounding boxes.
        frame = extract_text.tess_func(frame)

        # Find the Canny Edges.
        edges = cv2.Canny(frame, cannyThreshold1, cannyThreshold2)

        # Find straight lines.
        lines = cv2.HoughLines(edges, 1, np.pi/180, houghThreshold)

        # Draw them on the current frame/image.
        values = []
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                   if (theta < 1) and (len(lines) >=20):
                        print("Rho is {}, theta is {}".format(rho, theta))
                        x1, x2, y1, y2, x0, y0 = draw_lines.draw(rho, theta)
                        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)    
                        values.append([x1,x2,y1,y2, x0, y0])   

            for x1,x2,y1,y2,x0,y0 in values: 
                plt.scatter(x0,y0)  
            
            plt.show()

        # Update img to be have a rectangle around the uniform colours.
        low_colour_steelers, high_colour_steelers, low_colour_pats, high_colour_pats = colour_filter.uniform_colour_thresholds()
        mask_img = cv2.inRange(image, low_colour_steelers, high_colour_steelers) # Changed to image.
        img2 = draw_lines.outline_players(mask_img, values img2) # Deleted values.
        #mask_img2 = cv2.inRange(img, low_colour_pats, high_colour_pats)
        #mask_img2 = draw_lines.outline_players(mask_img2)
        
        # Create a combined video of Hough Line Transform result and the Canny edge detector.
        combined = np.concatenate((frame, img), axis=1)
        #combined2 = np.concatenate((mask_img, img2), axis=1)

        #combined_mask = np.concatenate((mask_img, mask_img2), axis=1)

        # Show the final output to the window. update output depending on filter used
        cv2.imshow('Play Example 1', img2)

        # Esc to quit.
        if cv2.waitKey(1) == 27:
            break 
            

# Call the function to run through.
filtering_main()

# Everything is done, release the video capture object.
cap.release()

# Close all the frames.
cv2.destroyAllWindows()

