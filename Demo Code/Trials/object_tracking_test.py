import cv2
import numpy as np

img = cv2.imread('Play_Ex/Play Capture 2.PNG')
lowerBound = np.array([50,15,65])
upperBound = np.array([60,60,100])

kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)

img = cv2.imread('Play_Ex/Play Capture 2.PNG')
img = cv2.resize(img,(340,220))

#convert BGR to HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# create the Mask
mask = cv2.inRange(imgHSV, lowerBound, upperBound)

cv2.imshow("mask",mask)
cv2.imshow("cam",img)
cv2.waitKey(10)
'''
#morphology
maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

maskFinal=maskClose
conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cv2.drawContours(img,conts,-1,(255,0,0),3)
for i in range(len(conts)):
    x,y,w,h=cv2.boundingRect(conts[i])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
    cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
cv2.imshow("maskClose",maskClose)
cv2.imshow("maskOpen",maskOpen)
cv2.imshow("mask",mask)
cv2.imshow("cam",img)
cv2.waitKey(10)'''