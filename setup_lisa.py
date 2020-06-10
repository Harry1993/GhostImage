import os
import string
import numpy as np
from PIL import Image

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

class LISAModel:
    num_channels = 3
    image_size = 32
    num_labels = 8

    def __init__(self, restore, session=None):

        model = Sequential()

        layers = [Conv2D(64, [8, 8],
                         strides=(2, 2),
                         padding="same",
                         input_shape=[self.image_size,
                                      self.image_size,
                                      self.num_channels],
                         data_format='channels_last'),
                  Activation('relu'),
                  Conv2D(64 * 2, [6, 6],
                         strides=(2, 2),
                         padding="valid"),
                  Activation('relu'),
                  Conv2D(64 * 2, [5, 5],
                         strides=(1, 1),
                         padding="valid"),
                  Activation('relu'),
                  Flatten(),
                  Dense(self.num_labels)]

        for layer in layers:
            model.add(layer)

        model.load_weights(restore)
        self.model = model

    def predict(self, data, tanhspace=0): # tanhspace doesn't matter here
        return self.model(data)

if __name__ == "__main__":
    data = LISA(mode='color', shuffle=False)
    print(data.data.shape, data.labels.shape)
