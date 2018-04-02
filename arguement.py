from PIL import Image
import os
import numpy as np
import tensorflow as tf
from scipy.misc import imsave

resize = tf.image.resize_images

root_A = "./input/horse2zebra/trainA"
root_B = "./input/horse2zebra/trainB"

filenames_A = os.listdir(root_A)
filenames_B = os.listdir(root_B)


with tf.Session() as sess:
    count = 0
    region = (42, 42, 214, 214)

    for name in filenames_A:
        path_A = os.path.join(root_A, name)
        try:
            img = Image.open(path_A).resize([256, 256])
            img_1 = sess.run(resize(np.array(img.rotate(45).crop(region)), size=[256, 256]))
            img_2 = sess.run(resize(np.array(img.rotate(-45).crop(region)), size=[256, 256]))
            img_3 = img.transpose(Image.FLIP_LEFT_RIGHT)
            if not os.path.exists("./input/horse2zebra/train_A/"):
                os.makedirs("./input/horse2zebra/train_A/")
            img.save("./input/horse2zebra/train_A/" + str(count) + "_0.jpg")
            imsave("./input/horse2zebra/train_A/" + str(count) + "_1.jpg", img_1)
            imsave("./input/horse2zebra/train_A/" + str(count) + "_2.jpg", img_2)
            img_3.save("./input/horse2zebra/train_A/" + str(count) + "_3.jpg")
        except:
            continue
        count += 1
        print(count)

    count = 0

    for name in filenames_B:
        path_B = os.path.join(root_B, name)
        try:
            img = Image.open(path_B).resize([256, 256])
            img_1 = sess.run(resize(np.array(img.rotate(45).crop(region).resize(img.size)), size=[256, 256]))
            img_2 = sess.run(resize(np.array(img.rotate(-45).crop(region).resize(img.size)), size=[256, 256]))
            img_3 = img.transpose(Image.FLIP_LEFT_RIGHT)
            if not os.path.exists("./input/horse2zebra/train_B/"):
                os.makedirs("./input/horse2zebra/train_B/")
            img.save("./input/horse2zebra/train_B/" + str(count) + "_0.jpg")
            imsave("./input/horse2zebra/train_B/" + str(count) + "_1.jpg", img_1)
            imsave("./input/horse2zebra/train_B/" + str(count) + "_2.jpg", img_2)
            img_3.save("./input/horse2zebra/train_B/" + str(count) + "_3.jpg")
        except:
            continue
        count += 1
        print(count)
