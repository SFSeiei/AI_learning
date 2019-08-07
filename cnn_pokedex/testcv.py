import cv2 as cv
from keras.preprocessing.image import img_to_array

# 读取图像，支持 bmp、jpg、png、tiff 等常用格式
img = cv.imread("examples/charmander_plush.png")
# print("img", img)
# print("imgShape", img.shape)
# print("img", img)
# 绝对地址能够显示
# "D:\Test\2.jpg"
# 创建窗口并显示图像
data = []
image = cv.resize(img, (96, 96))
# image = img_to_array(image)
output = image.copy()
print("output", image)
print("outputShape", image.shape)
data.append(image)

cv.namedWindow("Image")
cv.imshow("Image", img)
cv.waitKey(0)
# # 释放窗口
cv.destroyAllWindows()
# print(data)
cv.namedWindow("Image2")
cv.imshow("Image2", image)
cv.waitKey(0)
# # 释放窗口
cv.destroyAllWindows()
