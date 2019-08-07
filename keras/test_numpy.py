import numpy as np

ran = np.random.rand(2,3,4,5)
arr = np.arange(24).reshape(2, 3, 4)
arr1 = np.arange(4).reshape(4)
arr2 = np.array([[2, 1, 1], [1, 3, 2]])
arr3 = np.arange(12).reshape(3, 4)
print(ran)
print('max')
print(ran.max(axis=0))
#  返回一个可以看成是嵌套的矩阵。

# ran1 = np.random.randn(2, 3) * 1 +mu
#  返回一个服从标准正态分布的样本值矩阵。 后面的1为标准差（方差）o，后面mu是高斯分布的均值u。

# ran = np.random.normal(1, 2, (10,))
# 返回以1为中心(相当于高斯公式里的u)，2为scale，的正态分布的10个样本值。
# scale越小越瘦高，越大越矮胖，相当于高斯公式里的o.
# ran = np.random.normal(1, 0.05, (200,))
# print("原数：")
# print(ran)
# print("\n前160个：")
# print(ran[:160])
# print("\n后40个：")
# print(ran[160:])
