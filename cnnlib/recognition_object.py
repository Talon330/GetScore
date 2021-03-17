# -*- coding: utf-8 -*-
"""
识别图像的类，为了快速进行多次识别可以调用此类下面的方法：
R = Recognizer(image_height, image_width, max_captcha)
for i in range(10):
    r_img = Image.open(str(i) + ".jpg")
    t = R.rec_image(r_img)
简单的图片每张基本上可以达到毫秒级的识别速度
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.ops.gen_array_ops import expand_dims
from cnnlib.network import CNN
import json, os


class Recognizer(CNN):
    def __init__(self, image_height, image_width, max_captcha, char_set, model_save_dir):
        # 初始化变量
        super(Recognizer, self).__init__(max_captcha, char_set, 1.)

        model_save_dir = os.path.join(model_save_dir,'tf_model_weights.ckpt')

        self.build(input_shape=(None, 1, image_height, image_width))
        self.summary((1, image_height, image_width))
        self.load_weights(model_save_dir)

    def rec_image(self, img):
        # 读取图片
        img_array = np.array(img)
        test_image = self.convert2gray(img_array)
        test_image = test_image / 255
        for _ in range(4-len(test_image.shape)):
            test_image = np.expand_dims(test_image, 0)
        y_predict = self.predict(test_image)
        text_list = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)

        # 获取结果
        predict_text = text_list[0]
        p_text = ""
        for p in predict_text:
            p_text += str(self.char_set[p])

        # 返回识别结果
        return p_text


def main():
    with open("conf/sample_config.json", "r", encoding="utf-8") as f:
        sample_conf = json.load(f)
    image_height = sample_conf["image_height"]
    image_width = sample_conf["image_width"]
    max_captcha = sample_conf["max_captcha"]
    char_set = sample_conf["char_set"]
    model_save_dir = sample_conf["model_save_dir"]
    tf.keras.backend.clear_session()
    gpus= tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(gpus[0], True)
    R = Recognizer(image_height, image_width, max_captcha, char_set, model_save_dir)
    r_img = Image.open("./sample/test/_15889195303727891.jpg")
    t = R.rec_image(r_img)
    print(t)


if __name__ == '__main__':
    main()
