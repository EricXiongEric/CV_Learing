#导入包
import cv2 as cv
import numpy as np
#测试滤波
img = cv.imread("test.jpg",cv.IMREAD_GRAYSCALE)
print(img)
img1 = cv.blur(img,(3,3))#均值滤波
img2 = cv.boxFilter(img,-1,(3,3),normalize=False)#normalize 的意思是归一化，true就除，false就容易过界
img3 = cv.GaussianBlur(img,(3,3),1)
img4 = cv.medianBlur(img,3)#中值滤波，取中间值到G点
res = np.vstack((img1,img2,img3))#把所有结果拼接在一起 h为横着，v为竖着
cv.imshow('img',res)
cv.waitKey(0)
cv.destroyWindow()
#---------------------------形态学------------------------------
er = cv.erode(img,np.ones(3,3),iterations=1)#腐蚀操作,第二个参数为卷积核，第三为迭代次数
di = cv.dilate(img,np.ones(3,3),iterations=1)#膨胀操作，同上。与上面相反的过程
#-----------------开运算与必运算---------------------
#开运算，先腐蚀后膨胀
cv.morphologyEx(img,cv.MORPH_OPEN,np.ones(3,3))#最后的一个参数为卷积核
#闭运算，先膨胀后腐蚀
cv.morphologyEx(img,cv.MORPH_CLOSE,np.ones(3,3))
#--------------------------梯度运算----------------
#图像先膨胀 然后减去原图像腐蚀   就得到了图像的轮毂
cv.morphologyEx(img,cv.MORPH_GRADIENT,np.ones(3,3))
#--------------礼帽与黑帽-------------
#礼帽=原图像减去开运算（先腐蚀后膨胀），就留下毛刺了
cv.morphologyEx(img,cv.MORPH_TOPHAT,np.ones(3,3))
#黑帽=闭运算减去原式图像，就保留轮廓
cv.morphologyEx(img,cv.MORPH_BLACKHAT,np.ones(3,3))