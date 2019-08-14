import os
import numpy as np
from keras.models import Sequential, Model
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.utils.np_utils import to_categorical
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

imgs = []
labels = []
img_shape = (96, 96)
# image generator
files = os.listdir('../cnn_pokedex/examples')
# read 1000 files for the generator
for img_file in files[:6]:
    img = imread('../cnn_pokedex/examples/' + img_file).astype('float32')
    img = imresize(img, img_shape)
    imgs.append(img)

imgs = np.array(imgs)
train_gen = ImageDataGenerator(
    # rescale = 1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

val_gen = ImageDataGenerator(
    # rescale = 1./255,
    featurewise_center=True,
    featurewise_std_normalization=True)

train_gen.fit(imgs)
val_gen.fit(imgs)

# 4500 training images
train_iter = train_gen.flow_from_directory('../cnn_pokedex/dataset', class_mode='binary',
                                           target_size=img_shape, batch_size=16)
print("train_iter", train_iter)
# 501 validation images
val_iter = val_gen.flow_from_directory('../cnn_pokedex/examples', class_mode='binary',
                                       target_size=img_shape, batch_size=16)

img_iter = zip(train_iter, val_iter)
print("img_iter", img_iter)
# image generator debug
for x_batch, y_batch in img_iter:
    print(x_batch.shape)
    print(y_batch.shape)
    plt.imshow(x_batch[0])

# finetune from the base model VGG16
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
base_model.summary()

out = base_model.layers[-1].output
out = layers.Flatten()(out)
out = layers.Dense(1024, activation='relu')(out)
# 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
out = layers.Dropout(0.5)(out)
out = layers.Dense(512, activation='relu')(out)
out = layers.Dropout(0.3)(out)
out = layers.Dense(1, activation='sigmoid')(out)
tuneModel = Model(inputs=base_model.input, outputs=out)
tuneModel.summary()
for layer in tuneModel.layers[:19]:  # freeze the base model only use it as feature extractors
    layer.trainable = False
tuneModel.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

history = tuneModel.fit_generator(
    generator=train_iter,
    steps_per_epoch=100,
    epochs=100,
    validation_data=val_iter,
    validation_steps=32
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, 101)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.legend()
plt.show()
