from PIL import Image
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave

root = "/home/lwp/下载/horse"
horse = os.listdir(root)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    count = 0
    for img in horse:
        try:
            filename = os.path.join(root, img)
            image = np.array(Image.open(filename))
            resized_image = tf.image.resize_images(images=image,size=[256, 256], method=0)
            newimage = sess.run(resized_image)
            imsave(name="/home/lwp/下载/zebra/new_1_" + str(count) + ".jpg", arr=newimage)
            count += 1
            print(count)
        except:
            print(img)
