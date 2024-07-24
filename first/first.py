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

#----------------------算子操作-------------------
#sobel算子：分为x轴和y轴，对于x轴来说矩阵为假如为3*3 [[-1,0,1],-2,0,2,-1,0,1],相当于右边减左边，Y轴来说矩阵就向右旋转90度，就是下面减去上面。
img = cv.imread("xjpic.jpg",cv.IMREAD_GRAYSCALE)
res_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)#cv.CV_64F这个意思为精度更高，（1,0） 1表示为x轴，ksize为核大小
res_x = cv.convertScaleAbs(res_x)#把结果转换为绝对值
res_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
res_y = cv.convertScaleAbs(res_y)
res_add = cv.addWeighted(res_x,0.5,res_y,0.5,0)#建议分开操作，而不是（1,1），这样效果更好
totol_img = np.hstack((img,res_x,res_y,res_add))
#Scharr算子：跟sobel算子类似，只不过它的细节放大的更大，因为他的卷积核值要大些，对于x轴来说算子核为：假如为3*3 [[-3,0,3],[-10,0,10],[-3,0,3]],y轴同sobel算子一样
cv.Scharr(img,cv.CV_64F,1,0)#同上面sobel类似，只不过没有卷积核了
#laplacian算子：它就不分x轴和y轴了，他的核为[[0,1,0],[1,-4,1],[0,1,0]]
cv.Laplacian(img,cv.CV_64F)
