import cv2 as cv
import numpy as np
img = cv.imread("xjpic.jpg",cv.IMREAD_GRAYSCALE)
res_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
res_x = cv.convertScaleAbs(res_x)
res_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
res_y = cv.convertScaleAbs(res_y)
res_add = cv.addWeighted(res_x,0.5,res_y,0.5,0)
totol = np.hstack((img,res_x,res_y,res_add))

cv.imshow('img',totol)
cv.waitKey(0)
cv.destroyWindow()
