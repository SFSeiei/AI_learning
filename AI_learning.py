# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from matplotlib.image import imread


# -----------------------------------------DATE 2019.7.1----------------------------------------------------------
# 一.生成数据
# x =np.arange(0,6,0.1)
# y =np.sin(x)
# 绘制图形
# plt.plot(x,y)
# plt.show()

# -----------------------------------------DATE 2019.7.5----------------------------------------------------------
# 一.显示图像
# img = imread("D:/李昱达\DATA 800-999/7568_福原　愛/2019042319283275431200.jpg")
# plt.imshow(img)
# plt.show()

# 二.感知机：与门
# def AND(x1,x2):
#     w1,w2,theta = 0.5,0.5,0.7
#     tmp = w1*x1+w2*x2
#     if tmp <= theta:
#         print(0)
#     else:
#         print(1)
#
# AND(1,0)
# AND(1,1)
# AND(0,1)
# AND(0,0)

# b=偏置  theta=阈值   b=-theta
# x = np.array([0,1])
# w = np.array([0.5,0.5])
# b = -0.7 # theta= 0.7
# print(x*w)
# print(np.sum(x*w))
# print(np.sum(w*x)+b)

# 权重和偏置实现与门
# def AND(x1,x2):
#     x = np.array([x1,x2])
#     w = np.array([0.5,0.5])
#     b = -0.7
#     tmp = np.sum(x*w)+b
#     if tmp <= 0:
#         print(0)
#     else:
#         print(1)
# AND(1,0)
# AND(1,1)
# AND(0,1)
# AND(0,0)

# 感知机的局限性就在于它只能用一条直线来表示，曲线无法用感知机来表示
# 由曲线分割的空间称为非线性空间，由直线分割的空间称为线性空间

# 单层感知机=直线  多层感知机=曲线
# def XOR(x1, x2):    异或门实现    异或门是一种多层结构的神经网络（二层感知机）
#  s1 = NAND(x1, x2)   与非门
#  s2 = OR(x1, x2)      或门
#  y = AND(s1, s2)      与门
#  return y
#
# XOR(0, 0) # 输出0
# XOR(1, 0) # 输出1
# XOR(0, 1) # 输出1
# XOR(1, 1) # 输出0

# 通过实际上拥有权重的层数来表示神经网络的名称
# b=偏置，用于控制神经元被激活的容易程度
# w=权重，用于控制各个信号的重要性

# 激活函数 y=h（x）
# 表示：
# a = b+x1*w1+x2*w2
# y = h (a)

# “朴素感知机”是指单层网络，指的是激活函数使用了阶
# 跃函数A 的模型。“多层感知机”是指神经网络，即使用 sigmoid
# 函数（后述）等平滑的激活函数的多层网络。

# 以阈值为界，一旦超过阈值，就切换输出————称为”阶跃函数“。
# 感知机————以阶跃函数作为激活函数。
# 神经网络————以多种函数作为激活函数。

# 阶跃函数 numpy实现
# def step_function(x):
#     y= x>0
#     return y.astype(np.int)
#
# print(step_function(np.array([-0.1,2.1,-2,2])))

# 绘制图形
# def step_function(x):
#     return np.array(x>0,dtype=int)
#
# x = np.arange(-5.0,5.0,0.1)
# y = step_function(x)
#
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

# sigmoid函数并绘图
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
# x = np.arange(-5.0,5.0,0.1)
# y = sigmoid(x)
#
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()

# 感知机中神经元之间流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号。
# 神经网络的激活函数必须使用非线性函数。  线性 = 直线          非线性 = 曲线，折线

# ReLU函数
def ReLU(x):
    return np.maximum(0, x)


# -----------------------------------------DATE 2019.7.8----------------------------------------------------------

# 多维数组的定义：多维数组就是“数字的集合”，数字排成一列的集合、排成
# 长方形的集合、排成三维状或者（更加一般化的）N维状的集合都称为多维数
# 组。

# 一维数组
# A = np.array([1.0,2.0,3.0])
# print(A)
# print(np.ndim(A))
# print(A.shape)   #这里的shape是一个成员变量。  （3，） 是一个一维数组
#
# [ 1.  2.  3.]
# 1
# (3,)

# 二维数组
# B = np.array([[1.0,2.0],[2.0,3.0],[3.0,4.0]])
# print(B)
# print(np.ndim(B))
# print(B.shape)

# [[ 1.  2.]
#  [ 2.  3.]
#  [ 3.  4.]]
# 2          #表示二维数组（矩阵matrix）
# (3, 2)     #表示是一个3X2的矩阵，第一个维度对应第0维，第二个维度对应第1维。横向排列称为行（row），纵向排列称为列（column)

# 矩阵乘法
# A = np.array([[1,2,3],[4,5,6]])
# print(A.shape)
# B = np.array([[1,2],[3,4],[5,6]])
# print(B.shape)
# print("np.dot()")
# print(np.dot(A,B))            #用np.dot()方法来进行矩阵的乘法，因为矩阵的乘法又称为点乘。
# print(A*B)                    #error tips：operands could not be broadcast together with shapes (2,3) (3,2)
# （2*3）（3*2）的矩阵无法进行广播

# C = np.array([[1,2],[3,4]])
# D = np.array([[5,6],[7,8]])
# print("C*D:")
# print(C*D)                    #这个是算术乘法运算  #只有相同维数的矩阵才能进行广播。
# print("np.dot()")             #这个是矩阵点乘，需要第一个矩阵的第一维元素个数等于第二个矩阵的第零维元素个数
# print(np.dot(C,D))

# 神经网络
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 恒等函数
def identify_function(x):
    return x


#
# X = np.array([1.0,0.5])
# W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
# B1 = np.array([0.1,0.2,0.3])
#
# A1 = np.dot(X,W1)+B1
# print(A1)
# Z1 = sigmoid(A1)
# print(Z1)
#
# W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
# B2 = np.array([0.1,0.2])
#
# A2 = np.dot(A1,W2)+B2
# print(A2)
# Z2 = identify_function(A2)
# print(Z2)

# 代码总结
# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     network['b1'] = np.array([0.1, 0.2, 0.3])
#     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     network['b2'] = np.array([0.1, 0.2])
#     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
#     network['b3'] = np.array([0.1, 0.2])
#     return network
#
#
# def forward(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#
#     A1 = np.dot(x, W1) + b1
#     Z1 = sigmoid(A1)
#     A2 = np.dot(Z1, W2) + b2
#     Z2 = sigmoid(A2)
#     A3 = np.dot(Z2, W3) + b3
#     Z3 = identify_function(A3)
#
#     return Z3
#
#
# network = init_network()
# X = ([2.0, 1.5])
# Y = forward(network, X)
# print(Y)
# -----------------------------------------DATE 2019.7.26----------------------------------------------------------
# softmax()函数，处理分类问题时用到。
def softmax(a):  # 如输入为a = np.array([0.3, 2.9, 4.0])
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策，减去一个常数，最后结果不变。
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
    # 输出则为[ 0.01821127 0.24519181 0.73659691]，值的总和为1，因此每个值就是输入元素的概率，分别为1.8%,24.5%,73.6%
    # 可以说：“因为第2个元素的概率最高，所以答案是第2个类别。”
    # 可以回答：“有74 %的概率是第2个类别，有25 %的概率是第1个类别，有1 %的概率是第0个类别。”


# x = np.array([1.0, 2.0, 3.0])
# print(x.shape)  # (3,)

# f = np.load('./keras/classifier_data_file/mnist.npz')
# (x_train, t_train), (x_test, t_test) = (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])
# f.close()

# print('原数据结构：')
# print(x_train.shape)  # (60000, 28, 28)
# print(t_train.shape)  # (60000,)
# print(x_test.shape)  # (10000, 28, 28)
# print(t_test.shape)  # (10000,)

# import sys, os
#
# sys.path.append(os.pardir)
# from keras.utils import np_utils

#
# # import pickle
# # from PIL import Image
# #
# # from dataset.mnist import load_mnist
# # from mnist import load_mnist
#
# 数据变换
# 分为10个类别
# nb_classes = 10
#
# x_train_1 = x_train.reshape(60000, 784)
# # x_train_1 /= 255
# # x_train_1 = x_train_1.astype('float32')
# t_train_1 = np_utils.to_categorical(t_train, nb_classes)
# print('变换后的数据结构：')
# print(x_train_1.shape, t_train_1.shape)
#
# x_test_1 = x_test.reshape(10000, 784)
# t_test_1 = np_utils.to_categorical(t_test, nb_classes)
# print(x_test_1.shape, t_test_1.shape)


# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()
#
#
# img = x_train[0]
# label = t_train[0]
# print("label:")
# print(label)  # 5
#
# print("\nimg.shape:")
# print(img.shape)  # (784,)
# img = img.reshape(28, 28)  # 把图像的形状变成原来的尺寸
# print("\nimg.shape:")
# print(img.shape)  # (28, 28)
# img_show(img)

# def get_data():
#     (x_train, t_train), (x_test, t_test) = \
#         load_mnist(normalize=True, flatten=True, one_hot_label=False)
#     return x_test, t_test


# def init_network():
#     with open("sample_weight.pkl", 'rb') as f:
#         network = pickle.load(f)
#     return network
#
#
# def predict(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = softmax(a3)
#     return y
#
#
# x, t = x_test_1, y_test_1
# network = init_network()
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)  # 获取概率最高的元素的索引
#     if p == t[i]:
#         accuracy_cnt += 1
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 均方误差函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 设“2”为正确解
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 例1：“2”的概率最高的情况（0.6）
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# print(mean_squared_error(np.array(y1), np.array(t)))         #0.09750000000000003
# 例2：“7”的概率最高的情况（0.6）
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]


# print(mean_squared_error(np.array(y2), np.array(t)))             #0.5975

# 交叉熵误差函数(单个数据)
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))

# print(cross_entropy_error(np.array(y1), np.array(t)))           #0.510825457099338
# print(cross_entropy_error(np.array(y2), np.array(t)))           #2.302584092994546

# train_size = x_train_1.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train_1[batch_mask]
# t_batch = t_train_1[batch_mask]


# mini-batch、单个数据均可的交叉熵误差函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# 当监督函数是标签形式（非one-hot表示）
# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 数值微分函数
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


# 二次函数
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


# print(numerical_diff(function_1, 5))  # 0.1999999999990898
# print(numerical_diff(function_1, 10))  # 0.2999999999986347

# 绘制二次函数的图像
# x = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()


# 三维函数
def function_2(x):
    return x[0] ** 2 + x[1] ** 2
    # 或者return np.sum(x**2)


# 绘制三位函数的图像
# x0 = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
# x1 = np.arange(0.0, 20.0, 0.1)
# X = np.array([x0, x1])
# print(X)
# y = function_2(X)
# plt.xlabel("x")
# plt.ylabel("f(x)")
#
# ax = plt.axes(projection='3d')
# # ax.contour3D(x0, x1, y, 50, cmap='binary')
# # ax.plot3D(x0, x1, y, 'gray')
# # plt.show()
# ax.scatter3D(x0, x1, y, c=y, cmap='Greens')


# def f(x, y):
#     return x ** 2 + y ** 2
#
#
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
# ax.view_init(60, 35)

# -----------------------------------------DATE 2019.7.29----------------------------------------------------------
# 数值梯度函数1    该方法有误
# def numerical_gradient(f, x):
#     h = 1e-4  # 0.0001
#     grad = np.zeros_like(x)  # 生成和x形状相同的数组
#     print(x)
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         # f(x+h)的计算
#         x[idx] = tmp_val + h
#         fxh1 = f(x)
#         # f(x-h)的计算
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
#         grad[idx] = (fxh1 - fxh2) / (2 * h)
#         x[idx] = tmp_val  # 还原值
#     return grad

# 数值梯度函数2
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        # print("fxh1:",fxh1)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        # print("\nfxh2:", fxh2)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()
    return grad


# 梯度下降函数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


# 简单神经网络类
# class simpleNet:
#     def __init__(self):
#         self.W = np.random.randn(2, 3)  # 用高斯分布进行初始化
#
#     def predict(self, x):
#         return np.dot(x, self.W)
#
#     def loss(self, x, t):
#         z = self.predict(x)
#         y = softmax(z)
#         loss = cross_entropy_error(y, t)
#
#         return loss
#
#
# net = simpleNet()
# print('net.W:')
# print(net.W)
# x = np.array([0.6, 0.9])
# p = net.predict(x)
# # print(p)
# np.argmax(p)
# t = np.array([0, 0, 1])
#
#
# # print(net.loss(x, t))
#
#
# def f(W):
#     # print("x:", x, "\nt:", t)
#     return net.loss(x, t)
#
#
# # print(net.W.size)
#
# # for idx in range(net.W.size):
# #     tmp_val = net.W[idx]
# #     print(tmp_val)
#
#
# # f = lambda w: net.loss(x, t)  # lambda形式
#
# dW = numerical_gradient(f, net.W)
# print('gradient:')
# print(dW)


# 2层神经网络
# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size,
#                  weight_init_std=0.01):
#         # 初始化权重
#         self.params = {}
#         self.params['W1'] = weight_init_std * \
#                             np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = weight_init_std * \
#                             np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)
#
#     def predict(self, x):
#         W1, W2 = self.params['W1'], self.params['W2']
#         b1, b2 = self.params['b1'], self.params['b2']
#         a1 = np.dot(x, W1) + b1
#         z1 = sigmoid(a1)
#         a2 = np.dot(z1, W2) + b2
#         y = softmax(a2)
#         return y
#
#     # x:输入数据, t:监督数据
#     def loss(self, x, t):
#         y = self.predict(x)
#
#         return cross_entropy_error(y, t)
#
#     def accuracy(self, x, t):
#         y = self.predict(x)
#         y = np.argmax(y, axis=1)
#         t = np.argmax(t, axis=1)
#
#         accuracy = np.sum(y == t) / float(x.shape[0])
#         return accuracy
#
#     # x:输入数据, t:监督数据
#     def numerical_gradient(self, x, t):
#         # loss_W = lambda W: self.loss(x, t)
#         def loss_W():
#             return self.loss(x, t)
#
#         grads = {}
#         grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
#         grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
#         grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
#         grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
#
#         return grads


# train_loss_list = []
# 超参数
# iters_num = 10000
# train_size = x_train_1.shape[0]
# # print('train_size:', train_size)
# batch_size = 100
# learning_rate = 0.1
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# for i in range(iters_num):
#     # 获取mini-batch
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train_1[batch_mask]
#     t_batch = t_train_1[batch_mask]
#     # 计算梯度
#     grad = network.numerical_gradient(x_batch, t_batch)
#     # grad = network.gradient(x_batch, t_batch) # 高速版!
#     # 更新参数
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]
#     # 记录学习过程
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss)
#     print(train_loss_list)

# 改良版
# train_loss_list = []
# train_acc_list = []
# test_acc_list = []
# # 平均每个epoch的重复次数
# iter_per_epoch = max(train_size / batch_size, 1)
# # 超参数
# iters_num = 10000
# batch_size = 100
# learning_rate = 0.1
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# for i in range(iters_num):
#     # 获取mini-batch
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]
#
#     # 计算梯度
#     grad = network.numerical_gradient(x_batch, t_batch)
#     # grad = network.gradient(x_batch, t_batch) # 高速版!
#
#     # 更新参数
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]
#
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss)
#
#     # 计算每个epoch的识别精度
#     if i % iter_per_epoch == 0:
#         train_acc = network.accuracy(x_train, t_train)
#         test_acc = network.accuracy(x_test, t_test)
#         train_acc_list.append(train_acc)
#         test_acc_list.append(test_acc)
#         print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# -----------------------------------------DATE 2019.7.30----------------------------------------------------------

# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x  # 将x,y赋值到实例的下x,y中,则计算backward过程时，不用再额外接受x,y做参数。
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # 翻转x和y
        dy = dout * self.x

        return dx, dy


# # 购买苹果实例
# apple = 100
# apple_num = 2
# tax = 1.1
#
# # MulLayer
# mul_apple_layer = MulLayer()
# mul_tax_layer = MulLayer()
#
# # forward
# apple_price = mul_apple_layer.forward(apple, apple_num)
# price = mul_tax_layer.forward(apple_price, tax)
#
# print(price)  # 220
#
# # backward
# dprice = 1
# dapple_price, dtax = mul_tax_layer.backward(dprice)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# print(dapple, dapple_num, dtax)  # 2.2 110 200


# 加法层
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# 购买苹果和橘子的计算图例子
# apple = 100
# apple_num = 2
# orange = 150
# orange_num = 3
# tax = 1.1
# # layer
# mul_apple_layer = MulLayer()
# mul_orange_layer = MulLayer()
# add_apple_orange_layer = AddLayer()
# mul_tax_layer = MulLayer()
#
# # forward
# apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
# orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
# all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
# price = mul_tax_layer.forward(all_price, tax)  # (4)
#
# # backward
# dprice = 1
# dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
# dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
# dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)
# print(price)  # 715
# print(dapple_num, dapple, dorange, dorange_num, dtax)  # 110 2.2 3.3 165 650

# Relu(函数)层
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        # print('x<=0:', (x <= 0))
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# Sigmoid(函数)层
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / 1 + np.exp(-x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# Affine层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


# Softmax-with-Loss层
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax的输出
        self.t = None  # 监督数据（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


# 二层神经网络
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):

        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):

        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):

        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


# 梯度确认(gradient check)
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# x_batch = x_train_1[:3]
# t_batch = t_train_1[:3]
# grad_numerical = network.numerical_gradient(x_batch, t_batch)
# grad_backprop = network.gradient(x_batch, t_batch)
# # 求各个权重的绝对误差的平均值
# for key in grad_numerical.keys():
#     diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
#     print(key + ":" + str(diff))
#     # print('grad_backprop' + key + ":" + str(grad_backprop[key]))
#     # print('grad_numerical' + key + ":" + str(grad_numerical[key]))

# 误差反向传播法学习过程
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
# iters_num = 10000
# train_size = x_train_1.shape[0]
# batch_size = 100
# learning_rate = 0.1
# train_loss_list = []
# train_acc_list = []
# test_acc_list = []
# iter_per_epoch = max(train_size / batch_size, 1)
# for i in range(iters_num):
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train_1[batch_mask]
#     t_batch = t_train_1[batch_mask]
#
#     # 通过误差反向传播法求梯度
#     grad = network.gradient(x_batch, t_batch)
#     # 更新
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]
#
#     # print(grad)
#
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss)
#
#     if i % iter_per_epoch == 0:
#         train_acc = network.accuracy(x_train_1, t_train_1)
#         test_acc = network.accuracy(x_test_1, t_test_1)
#         train_acc_list.append(train_acc)
#         test_acc_list.append(test_acc)
#         print('train_acc', train_acc, '\ntest_acc', test_acc)
#         # print('train_acc_list', train_acc_list, '\ntest_acc_list', test_acc_list)


# -----------------------------------------DATE 2019.7.30----------------------------------------------------------
# SGD优化器函数
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# Momentum(动量)优化器函数
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):

        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):

        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# AdaGrad优化器函数
class AdaGrad:
    def __init__(self, lr=0.01):

        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# 隐藏层的激活值
# x = np.random.randn(1000, 100)  # 1000个数据
# node_num = 100  # 各隐藏层的节点（神经元）数
# hidden_layer_size = 5  # 隐藏层有5层
# activations = {}  # 激活值的结果保存在这里
#
# for i in range(hidden_layer_size):
#     if i != 0:
#         x = activations[i - 1]
#     # w = np.random.randn(node_num, node_num) * 1
#     node_num = 100  # 前一层的节点数
#     w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
#     z = np.dot(x, w)
#     a = sigmoid(z)  # sigmoid函数
#     activations[i] = a
#
# # for i in range(hidden_layer_size):
# #     if i != 0:
# #         x = activations[i - 1]
# #     node_num = 100  #
# #     w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num)
# #     z = np.dot(x, w)
# #     a = ReLU(z)
# #     activations[i] = a
#
# # 绘制直方图
# for i, a in activations.items():
#     plt.subplot(1, len(activations), i + 1)
#     plt.title(str(i + 1) + "-layer")
#     plt.hist(a.flatten(), 30, range=(0, 1))
# plt.show()

# 过拟合
# network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100,
#                                                           100, 100, 100], output_size=10)
# optimizer = SGD(lr=0.01)  # 用学习率为0.01的SGD更新参数
# max_epochs = 201
# train_size = x_train.shape[0]
# batch_size = 100
# train_loss_list = []
# train_acc_list = []
# test_acc_list = []
# iter_per_epoch = max(train_size / batch_size, 1)
# epoch_cnt = 0
# for i in range(1000000000):
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]
#     grads = network.gradient(x_batch, t_batch)
#     optimizer.update(network.params, grads)
#     if i % iter_per_epoch == 0:
#         train_acc = network.accuracy(x_train, t_train)
#     test_acc = network.accuracy(x_test, t_test)
#     train_acc_list.append(train_acc)
#     test_acc_list.append(test_acc)
#     epoch_cnt += 1
#     if epoch_cnt >= max_epochs:
#         break

# -----------------------------------------DATE 2019.8.1----------------------------------------------------------
# im2col (image to column)函数例子
# import sys, os
#
# sys.path.append(os.pardir)
# from common.util import im2col
#
# x1 = np.random.rand(1, 3, 7, 7)
# col1 = im2col(x1, 5, 5, stride=1, pad=0)
# print(col1.shape)  # (9, 75)
# x2 = np.random.rand(10, 3, 7, 7)  # 10个数据
# col2 = im2col(x2, 5, 5, stride=1, pad=0)
# print(col2.shape)  # (90, 75)

# Convolution(卷积层)
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T  # 滤波器的展开
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out


# pooling池化层
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 展开(1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # 最大值(2)
        out = np.max(col, axis=1)
        # 转换(3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out


# 简单卷积神经网络
class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5,
                             'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / \
                           filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) *
                               (conv_output_size / 2))
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 设定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        return grads

# -----------------------------------------DATE 2019.8.2----------------------------------------------------------
