import os
import numpy as np
import tensorflow as  tf
from main import Img2ImgGAN
from scipy.misc import imsave

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

ckpt_dir = "./output/checkpoint"  # 检查点路径
sample_path_A = "./sample/horse2zebra/testA"
sample_path_B = "./sample/horse2zebra/testB"

model = Img2ImgGAN()
model.model_setup()
saver = tf.train.Saver()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session(config=config) as sess:
    sess.run(init)
    ckpt_name = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_name is not None:
        saver.restore(sess, ckpt_name)
    for name in os.listdir(sample_path_A):
        path_A = os.path.join(sample_path_A, name)
        img_A = model.read_img(path_A)
        fake_B_temp = sess.run(
            [model.fake_B],
            feed_dict={
                model.input_A: img_A}
        )
        imsave("./sample/fakeB_" + name[7:], ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
        imsave("./sample/" + name, ((img_A[0] + 1) * 127.5).astype(np.uint8))
    for name in os.listdir(sample_path_B):
        path_B = os.path.join(sample_path_B, name)
        img_B = model.read_img(path_B)
        fake_A_temp = sess.run(
            [model.fake_A],
            feed_dict={
                model.input_B: img_B}
        )
        imsave("./sample/fakeA_" + name[7:], ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
        imsave("./sample/" + name, ((img_B[0] + 1) * 127.5).astype(np.uint8))
