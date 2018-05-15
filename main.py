import numpy as np
from scipy.misc import imsave
from PIL import Image
import random
from model import discriminator
from model import generator
import tensorflow as tf
import os
import sys

to_train = True  # 是否训练
to_test = True  # 是否测试
to_restore = True  # 是否存储检查点（参数）
log_dir = "./output/log"  # 可视化日志路径
ckpt_dir = "./output/checkpoint"  # 检查点路径

max_images = 1000  # 数组中最多存储的训练/测试数据（batch_size, img_height, img_width, img_layer）数目
pool_size = 50  # 用于更新D的假图像的批次数
max_epoch = 100  # 每次训练的epoch数目
n_critic = 5  # 判别器训练的次数

img_height = 256  # 图像高度
img_width = 256  # 图像宽度
img_layer = 3  # 图像通道
batch_size = 1  # 一个批次的数据中图像的个数

save_training_images = True  # 是否存储训练数据

root_A = "./input/horse2zebra/trainA"
root_B = "./input/horse2zebra/trainB"
test_root_A = "./input/horse2zebra/testA"
test_root_B = "./input/horse2zebra/testB"


class Img2ImgGAN():
    def model_setup(self):
        '''
        build the model
        :return: None
        '''
        self.fake_images_A = np.zeros((pool_size, batch_size, img_height, img_width, img_layer))
        self.fake_images_B = np.zeros((pool_size, batch_size, img_height, img_width, img_layer))

        self.input_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer])
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer])

        self.fake_pool_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer])
        self.fake_pool_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer])

        self.num_fake_inputs = 0

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("img2img") as scope:
            self.scope = scope

            self.fake_B = generator(self.input_A, name="g_A")
            self.fake_A = generator(self.input_B, name="g_B")
            self.rec_A = discriminator(self.input_A, "d_A")
            self.rec_B = discriminator(self.input_B, "d_B")

            scope.reuse_variables()

            self.fake_rec_A = discriminator(self.fake_A, "d_A")
            self.fake_rec_B = discriminator(self.fake_B, "d_B")
            self.cyc_A = generator(self.fake_B, "g_B")
            self.cyc_B = generator(self.fake_A, "g_A")

            scope.reuse_variables()

            self.fake_pool_rec_A = discriminator(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = discriminator(self.fake_pool_B, "d_B")

    def loss_calc(self):
        '''
        build the loss function
        :return: None
        '''
        ####################
        # cycle loss
        ####################
        cyc_loss_A = tf.reduce_mean(tf.abs(self.input_A - self.cyc_A))
        cyc_loss_B = tf.reduce_mean(tf.abs(self.input_B - self.cyc_B))

        ####################
        # standard generator loss of g_A and g_B
        ####################
        gen_loss_A = tf.reduce_mean((self.fake_rec_B - 1) ** 2)
        gen_loss_B = tf.reduce_mean((self.fake_rec_A - 1) ** 2)

        ####################
        # discriminator loss with gradient penalty of d_B
        ####################
        disc_loss_B = (tf.reduce_mean(self.fake_pool_rec_B ** 2) + tf.reduce_mean((self.rec_B - 1) ** 2)) / 2.0

        ####################
        # discriminator loss with gradient penalty of d_A
        ####################
        disc_loss_A = (tf.reduce_mean(self.fake_pool_rec_A ** 2) + tf.reduce_mean((self.rec_A - 1) ** 2)) / 2.0

        self.g_loss_A = cyc_loss_A * 10 + cyc_loss_B * 10 + gen_loss_A  # g_A的损失函数
        self.g_loss_B = cyc_loss_A * 10 + cyc_loss_B * 10 + gen_loss_B  # g_B的损失函数
        self.d_loss_A = disc_loss_A  # d_A的损失函数
        self.d_loss_B = disc_loss_B  # d_B的损失函数

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(self.d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(self.d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(self.g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(self.g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: print(var.name)

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", self.g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", self.g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", self.d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", self.d_loss_B)

    def save_training_images(self, sess, epoch, A_input, B_input):
        '''
        save the training images
        :param sess: current session
        :param epoch: the epoch number
        :param A_input: sample from set A
        :param B_input: sample from set B
        :return:
        '''
        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")
        for i in range(0, 10):
            path_A = os.path.join(root_A, A_input[i])
            path_B = os.path.join(root_B, B_input[i])
            try:
                img_A = np.array(Image.open(path_A)).reshape([batch_size, img_height, img_width, img_layer]) / 127.5 - 1
                img_B = np.array(Image.open(path_B)).reshape([batch_size, img_height, img_width, img_layer]) / 127.5 - 1
            except:
                print(path_A)
                print(path_B)
                print("Can not open this image, skip this iteration,", sys._getframe().f_code.co_name)
                continue
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                [self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                feed_dict={self.input_A: img_A, self.input_B: img_B}
            )
            imsave("./output/imgs/fakeA_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/fakeB_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/cycA_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((cyc_A_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/cycB_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((cyc_B_temp[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/inputA_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((img_A[0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/inputB_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((img_B[0] + 1) * 127.5).astype(np.uint8))

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        '''
        pool of the fake images
        :param num_fakes: number of images in the pool
        :param fake: current fake image
        :param fake_pool: the fake images pool
        :return: tensor
        '''
        if num_fakes < pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        '''
        train the model
        :return: None
        '''
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        print("Build the network...")
        self.model_setup()
        print("Loss function calculations...")
        self.loss_calc()
        print("Load training data from the dataset folder...")
        A_input = os.listdir(root_A)
        B_input = os.listdir(root_B)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("The log writer...")
            writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
            print("Initializing the global variables...")
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            print("Restore the model to run it from last checkpoint...")
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(ckpt_dir)
                if chkpt_fname is not None:
                    saver.restore(sess, chkpt_fname)
            print("Training Loop...")
            for epoch in range(0, max_epoch):
                print("In the epoch ", epoch)
                # 按照条件调整学习率
                curr_lr = 2e-4 - epoch * 2e-6
                # 打乱输入 A 与输入 B 的对应顺序
                random.shuffle(A_input)
                random.shuffle(B_input)
                # 保存生成的图像
                if (save_training_images):
                    print("Save the training images...")
                    self.save_training_images(sess, epoch, A_input, B_input)
                for ptr in range(0, max_images):
                    print("In the iteration ", ptr)
                    path_A = os.path.join(root_A, A_input[ptr])
                    path_B = os.path.join(root_B, B_input[ptr])
                    try:
                        img_A = np.array(Image.open(path_A)).reshape(
                            [batch_size, img_height, img_width, img_layer]) / 127.5 - 1
                        img_B = np.array(Image.open(path_B)).reshape(
                            [batch_size, img_height, img_width, img_layer]) / 127.5 - 1
                    except:
                        print(path_A)
                        print(path_B)
                        print("Can not open this image, skip this iteration,", sys._getframe().f_code.co_name)
                        continue
                    # Optimizing the G_A network
                    _, summary_str = sess.run(
                        [self.g_A_trainer, self.g_A_loss_summ],
                        feed_dict={
                            self.input_A: img_A,
                            self.input_B: img_B,
                            self.lr: curr_lr}
                    )
                    writer.add_summary(summary_str, epoch * max_images + ptr)
                    # Optimizing the D_B network
                    for i in range(n_critic):
                        fake_B = sess.run(self.fake_B, feed_dict={self.input_A: img_A})
                        fake_B_temp = self.fake_image_pool(self.num_fake_inputs, fake_B, self.fake_images_B)
                        _, summary_str = sess.run(
                            [self.d_B_trainer, self.d_B_loss_summ],
                            feed_dict={
                                self.input_A: img_A,
                                self.input_B: img_B,
                                self.lr: curr_lr,
                                self.fake_pool_B: fake_B_temp}
                        )
                    writer.add_summary(summary_str, epoch * max_images + ptr)
                    # Optimizing the G_B network
                    _, summary_str = sess.run(
                        [self.g_B_trainer, self.g_B_loss_summ],
                        feed_dict={
                            self.input_A: img_A,
                            self.input_B: img_B,
                            self.lr: curr_lr}
                    )
                    writer.add_summary(summary_str, epoch * max_images + ptr)
                    # Optimizing the D_A network
                    for i in range(n_critic):
                        fake_A = sess.run(self.fake_A, feed_dict={self.input_B: img_B})
                        fake_A_temp = self.fake_image_pool(self.num_fake_inputs, fake_A, self.fake_images_A)
                        _, summary_str = sess.run(
                            [self.d_A_trainer, self.d_A_loss_summ],
                            feed_dict={
                                self.input_A: img_A,
                                self.input_B: img_B,
                                self.lr: curr_lr,
                                self.fake_pool_A: fake_A_temp}
                        )
                    writer.add_summary(summary_str, epoch * max_images + ptr)

                    self.num_fake_inputs += 1
                print("Save the model...")
                saver.save(sess, os.path.join(ckpt_dir, "img2img"), global_step=epoch)

    def test(self):
        '''
        test the model
        :return: None
        '''
        self.model_setup()
        print("Load test data from the dataset folder...")
        A_input = os.listdir(test_root_A)
        B_input = os.listdir(test_root_B)
        saver = tf.train.Saver()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init)
            chkpt_fname = tf.train.latest_checkpoint(ckpt_dir)
            print("Restore the model...")
            saver.restore(sess, chkpt_fname)
            if not os.path.exists("./output/test/"):
                os.makedirs("./output/test/")
            print("Testing loop...")
            for i in range(0, min(len(A_input), len(B_input))):
                print("In the iteration ", i)
                path_A = os.path.join(test_root_A, A_input[i])
                path_B = os.path.join(test_root_B, B_input[i])
                try:
                    img_A = np.array(Image.open(path_A)).reshape(
                        [batch_size, img_height, img_width, img_layer]) / 127.5 - 1
                    img_B = np.array(Image.open(path_B)).reshape(
                        [batch_size, img_height, img_width, img_layer]) / 127.5 - 1
                except:
                    print(path_A)
                    print(path_B)
                    print("Can not open this image, skip this iteration,", sys._getframe().f_code.co_name)
                    continue
                fake_A_temp, fake_B_temp = sess.run(
                    [self.fake_A, self.fake_B],
                    feed_dict={
                        self.input_A: img_A,
                        self.input_B: img_B}
                )
                imsave("./output/test/fakeA_" + str(i) + ".jpg", ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/test/fakeB_" + str(i) + ".jpg", ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/test/inputA_" + str(i) + ".jpg", ((img_A[0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/test/inputB_" + str(i) + ".jpg", ((img_B[0] + 1) * 127.5).astype(np.uint8))


def main():
    if to_train:
        train = Img2ImgGAN()
        train.train()
    if to_test:
        test = Img2ImgGAN()
        test.test()


if __name__ == '__main__':
    main()
