import cv2 as cv
import numpy as np
img = cv.imread("bro.jpg",cv.IMREAD_GRAYSCALE,)
img = cv.resize(img,None,fx=0.2,fy=0.2)
res_x = cv.Scharr(img,cv.CV_64F,1,0)
res_x = cv.convertScaleAbs(res_x)
res_y = cv.Scharr(img,cv.CV_64F,0,1)
res_y = cv.convertScaleAbs(res_y)
res_add = cv.addWeighted(res_x,0.5,res_y,0.5,0)
totol = np.hstack((img,res_x,res_y,res_add))

cv.imshow('img',res_add)
cv.waitKey(0)
cv.destroyWindow()
