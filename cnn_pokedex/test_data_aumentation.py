# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessory packages
from keras.preprocessing.image import ImageDataGenerator  # 用于数据增强
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split  # 用来创建训练和测试分叉。
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np

import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e.,directory of images)")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for ,initial learning rate,
# batch_size , and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (224, 224, 3)  # (96,96,3)

# initialize the data and labels
data = []
labels = []
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
# print(imagePaths)
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image , pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    # print("image", image)
    # print("imagePath", imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    print("imagePath.split", imagePath.split(os.path.sep))
    print("label", label)
    labels.append(label)

# scale the raw pixel intensities to the range
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80%
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation(数据增强)
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
# initialize the model
# print("[INFO] compiling model...")
# model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb.classes_))
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

testX_aug = aug.flow(testX, testY, batch_size=BS)

x_batch = testX_aug.next()
plt.imshow(x_batch[0].reshape(1024, 2058), cmap='gray')
plt.show()

# # train the network
# print("[INFO] training network...")
# H = model.fit_generator(
#     aug.flow(testX, testY, batch_size=BS),
#     validation_data=(testX, testY),
#     steps_per_epoch=len(trainX) // BS,
#     epochs=EPOCHS, verbose=1)
