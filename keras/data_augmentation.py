# from keras.preprocessing.image import ImageDataGenerator
# from keras.datasets import mnist
# from keras.datasets import cifar10
# from keras.utils import np_utils
# import numpy as np
# import matplotlib.pyplot as plt
#
# num_classes = 10
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = np.expand_dims(x_train, axis=3)
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)
#
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
#
# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(x_train)
# plt.subplot(2, 4, 1)
# plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
# plt.show()
# data_iter = datagen.flow(x_train, y_train, batch_size=8)
#
# x_batch, y_batch = data_iter.next()
# print("data_iter.next()", data_iter.next())
# print("x_batch", x_batch, "\ny_batch", y_batch)
# plt.subplot(2, 4, 1)
# plt.imshow(x_batch[0].reshape(28, 28), cmap='gray')
# plt.show()

# ___________________Test local images augmentation_______________
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10
seed = 1
imagegen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
maskgen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
image_iter = imagegen.flow_from_directory('../cnn_pokedex/dataset/bulbasaur', target_size=(224, 224), class_mode=None,
                                          batch_size=10,
                                          seed=seed)
mask_iter = maskgen.flow_from_directory('../cnn_pokedex/dataset/charmander', color_mode='rgb', target_size=(224, 224),
                                        class_mode=None,
                                        batch_size=10, seed=seed)

data_iter = zip(image_iter, mask_iter)
# data_iter = zip(image_iter)
while True:
    for x_batch, y_batch in data_iter:
        for i in range(8):
            print(i // 4)
            plt.subplot(2, 8, i + 1)
            plt.imshow(x_batch[i].reshape(224, 224, 3))
            plt.subplot(2, 8, 8 + i + 1)
            plt.imshow(y_batch[i].reshape(224, 224, 3), cmap='gray')
        plt.show()
