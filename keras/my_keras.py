from keras.models import Sequential
from keras.layers import Dense
import keras

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(100,)))  # 第一个参数是batch_size,第二个input_shape=(784,)是输入
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])  # 配置学习过程，第一个optimizer是优化器，第二个loss是损失函数，第三个metrics是评估标准。

# 生成虚拟数据
import numpy as np

data = np.random.random((1000, 100))
test_data = np.random.random((1000, 100))

labels = np.random.randint(2, size=(1000, 1))
test_labels = np.random.randint(2, size=(1000, 1))

# 将标签转换为分类的one-hot编码
# one_hot_labels = keras.utils.to_categorical(labels,num_classes=10)
# model.fit(data,one_hot_labels,epochs=10,batch_size=32)

print("______Training_______")
model.fit(data, labels, epochs=10, batch_size=32)

print("\n_________Test________")
loss, accuracy = model.evaluate(test_data, test_labels)

print("test loss:", loss)
print("\ntest accuracy", accuracy)
