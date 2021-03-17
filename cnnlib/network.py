import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random
from tensorflow.keras import layers, models
from tensorflow.python.ops.nn_ops import dropout

class CNN(models.Model):
    def __init__(self, max_captcha, char_set, droprate):
        super(CNN, self).__init__()
        # 初始值
        self.max_captcha = max_captcha
        self.char_set = char_set
        self.char_set_len = len(char_set)
        with tf.name_scope('parameters'):
            self.w_alpha = 0.01
            self.b_alpha = 0.1
            self.keep_prob = droprate  # dropout值

    @staticmethod
    def convert2gray(img):
        """
        图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
        :param img:
        :return:
        """
        if len(img.shape) == 3 and img.shape[0] == 3:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return np.expand_dims(gray, 0)
        elif len(img.shape) == 2:
            return np.expand_dims(img, 0)
        else:
            return img

    def text2vec(self, text):
        """
        转标签为oneHot编码
        :param text: str
        :return: numpy.array
        """
        text_len = len(text)
        if text_len > self.max_captcha:
            raise ValueError('验证码最长{}个字符'.format(self.max_captcha))

        vector = np.zeros(self.max_captcha * self.char_set_len)

        for i, ch in enumerate(text):
            idx = i * self.char_set_len + self.char_set.index(ch)
            vector[idx] = 1
        return vector
    
    def build(self, input_shape):
        # 卷积层1
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=True, activation='relu', kernel_initializer='glorot_normal',bias_initializer='random_normal')
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same')
        self.drop1 = layers.Dropout(self.keep_prob)

        # 卷积层2
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=True, activation='relu', kernel_initializer='glorot_normal',bias_initializer='random_normal')
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same')
        self.drop2 = layers.Dropout(self.keep_prob)

        # 卷积层3
        self.conv3 = layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=True, activation='relu', kernel_initializer='glorot_normal',bias_initializer='random_normal')
        self.pool3 = layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same')
        self.drop3 = layers.Dropout(self.keep_prob)

        self.flatten = layers.Flatten()

        # 全连接层1
        self.dense1 = layers.Dense(1024,activation='relu', kernel_initializer='glorot_normal',bias_initializer='random_normal')
        self.drop4 = layers.Dropout(self.keep_prob)

        # 全连接层2
        self.dense2 = layers.Dense(self.max_captcha * self.char_set_len, kernel_initializer='glorot_normal',bias_initializer='random_normal')
        self.drop5 = layers.Dropout(self.keep_prob)
        super(CNN, self).build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop4(x)
        x = self.dense2(x)
        x = self.drop5(x)
        return x

    def summary(self, shape):
        x_input = layers.Input(shape=shape)
        output = self.call(x_input)
        model = tf.keras.Model(inputs = x_input, outputs = output)
        model.summary()