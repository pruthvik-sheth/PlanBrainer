import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt


class DataLoader:

    def __init__(self) -> None:
        pass

    @staticmethod
    def test(path):
        sample_img = tf.io.read_file(path)
        sample_img = tf.io.decode_png(sample_img)
        print(sample_img.shape)
        # plt.figure()
        # plt.imshow(sample_img)
        # plt.show()
        

    @staticmethod
    def load_image(image_file):
        """Loading and splitting a single image using tf"""
        image = tf.io.read_file(image_file)
        image = tf.io.decode_png(image)

        img_width = tf.shape(image)[1]
        img_width = img_width // 2

        # Splitting the image
        real_image = image[:, img_width:, :]
        input_image = image[:, :img_width, :]

        print(real_image)

        # Casting to float 32
        real_image = tf.cast(real_image, tf.float32)
        input_image = tf.cast(input_image, tf.float32)

        return real_image, input_image
    
    @staticmethod
    def _resize_image(input_image, real_image, height, width):
        input_image = tf.image.resize(
            input_image,
            [height, width],
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        real_image = tf.image.resize(
            real_image,
            [height, width],
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return real_image, input_image
    
    @staticmethod
    def _random_crop(input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis = 0)
        cropped_image = tf.image.random_crop(
            stacked_image,
            size = [2, 512, 512, 3]
        )

    @staticmethod
    def preprocess_data():
        pass