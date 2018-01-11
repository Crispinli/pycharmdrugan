'''
算法描述：
    （1）模型整体结构：
        a. 整体结构类似 CycleGAN 模型，并且进行了改进
        b. 模型中的包含两个 GAN 模型，并同时进行优化
        c. 两个 GAN 当中的生成器 generator 和判别器 discriminator 的结构相同
        d. 对每个 GAN 的判别器进行 1 次优化，然后对生成器进行 1 次优化
    （2）生成器 generator 的结构：
        a. 整体结构类似 U-Net 模型的形式，并且进行了改进
        b. 在模型的 bottom 部分，包含 9 个残差块
        c. 在 encoder 部分，编码结果直接与 decoder 部分的对应结果进行拼接
    （3）判别器 discriminator 结构：
        a. 整体结构为全卷积网络 FCN 的形式
        b. 输出是一个经过编码操作的图像块
        c. 输入是全图像的形式，尺寸为 [1, 256, 256, 3]
    （4）模型的损失函数：
        a. 两个 GAN 的损失函数具有相同的形式
        b. 损失函数类似 WGAN_GP 的形式，并且进行了改进
        c. 判别器损失的计算方式不变，在生成器损失中加入 cycle loss 项
    （5）模型训练策略：
        a. 最优化算法采用 tf.train.AdamOptimizer 算法
        b. 一次训练会进行 20 个 epoch，每个 epoch 中进行 1000 次迭代
        c. 学习率 2e-4，每进行一个 epoch 的训练，学习率减少 1e-5
'''
import numpy as np
from scipy.misc import imsave
import os
import random
from model import discriminator
from model import generator
import tensorflow as tf

to_train = True  # 是否训练
to_test = True  # 是否测试
to_restore = True  # 是否存储检查点（参数）
log_dir = "./output/log"  # 可视化日志路径
ckpt_dir = "./output/checkpoint"  # 检查点路径

max_images = 1000  # 数组中最多存储的训练/测试数据（batch_size, img_height, img_width, img_layer）数目
pool_size = 50  # 用于更新D的假图像的批次数
max_epoch = 20  # 每次训练的epoch数目
n_critic = 1  # 判别器训练的次数

img_height = 256  # 图像高度
img_width = 256  # 图像宽度
img_layer = 3  # 图像通道
img_size = img_height * img_width  # 图像尺寸
batch_size = 1  # 一个批次的数据中图像的个数

save_training_images = True  # 是否存储训练数据


class DRUGAN():
    def input_setup(self):

        # 获取图像的名字，得到文件名列表
        self.filenames_A = tf.train.match_filenames_once("./input/horse2zebra/trainA/*.jpg")
        self.filenames_B = tf.train.match_filenames_once("./input/horse2zebra/trainB/*.jpg")

        # 把文件名列表转换成队列
        filename_queue_A = tf.train.string_input_producer(self.filenames_A)
        filename_queue_B = tf.train.string_input_producer(self.filenames_B)

        # 从队列中读取图像
        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        # 转换图像格式，并做灰度处理
        self.image_A = tf.subtract(
            tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [256, 256]), 127.5), 1)
        self.image_B = tf.subtract(
            tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [256, 256]), 127.5), 1)

    def input_read(self, sess):

        # Loading images into the tensors
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        self.fake_images_A = np.zeros((pool_size, 1, img_height, img_width, img_layer))
        self.fake_images_B = np.zeros((pool_size, 1, img_height, img_width, img_layer))

        self.A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
        self.B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))

        for i in range(max_images):
            image_tensor = sess.run(self.image_A)
            if (image_tensor.size == img_size * batch_size * img_layer):
                self.A_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))

        for i in range(max_images):
            image_tensor = sess.run(self.image_B)
            if (image_tensor.size == img_size * batch_size * img_layer):
                self.B_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))

        coord.request_stop()
        coord.join(threads)

    def model_setup(self):

        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")

        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")

        self.num_fake_inputs = 0

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("drugan") as scope:
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

        ####################
        # cycle loss
        ####################
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A - self.cyc_A) + tf.abs(self.input_B - self.cyc_B))

        ####################
        # standard generator loss of g_A and g_B
        ####################
        gen_loss_A = -tf.reduce_mean(self.fake_rec_B)
        gen_loss_B = -tf.reduce_mean(self.fake_rec_A)

        ####################
        # discriminator loss with gradient penalty of d_B
        ####################
        disc_loss_B = tf.reduce_mean(self.fake_pool_rec_B) - tf.reduce_mean(self.rec_B)
        alpha_B = tf.random_uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
        interpolates_B = self.input_B + alpha_B * (self.fake_B - self.input_B)
        with tf.variable_scope(self.scope) as scope_B:
            scope_B.reuse_variables()
            gradients_B = tf.gradients(discriminator(interpolates_B, name="d_B"), [interpolates_B])[0]
        slopes_B = tf.sqrt(tf.reduce_sum(tf.square(gradients_B), reduction_indices=[1]))
        gradients_penalty_B = tf.reduce_mean((slopes_B - 1.0) ** 2)
        disc_loss_B += 10 * gradients_penalty_B

        ####################
        # discriminator loss with gradient penalty of d_A
        ####################
        disc_loss_A = tf.reduce_mean(self.fake_pool_rec_A) - tf.reduce_mean(self.rec_A)
        alpha_A = tf.random_uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
        interpolates_A = self.input_A + alpha_A * (self.fake_A - self.input_A)
        with tf.variable_scope(self.scope) as scope_A:
            scope_A.reuse_variables()
            gradients_A = tf.gradients(discriminator(interpolates_A, name="d_A"), [interpolates_A])[0]
        slopes_A = tf.sqrt(tf.reduce_sum(tf.square(gradients_A), reduction_indices=[1]))
        gradients_penalty_A = tf.reduce_mean((slopes_A - 1.0) ** 2)
        disc_loss_A += 10 * gradients_penalty_A

        self.g_loss_A = cyc_loss * 10 + gen_loss_A  # g_A的损失函数
        self.g_loss_B = cyc_loss * 10 + gen_loss_B  # g_B的损失函数
        self.d_loss_A = disc_loss_A  # d_A的损失函数
        self.d_loss_B = disc_loss_B  # d_B的损失函数

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.99)

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

    def save_training_images(self, sess, epoch):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0, 10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                [self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                feed_dict={self.input_A: self.A_input[i], self.input_B: self.B_input[i]}
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
                   ((self.A_input[i][0] + 1) * 127.5).astype(np.uint8))
            imsave("./output/imgs/inputB_" + str(epoch) + "_" + str(i) + ".jpg",
                   ((self.B_input[i][0] + 1) * 127.5).astype(np.uint8))

    def fake_image_pool(self, num_fakes, fake, fake_pool):

        if (num_fakes < pool_size):
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

        ''' Training Function '''

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("Load Dataset from the dataset folder...")
        self.input_setup()
        print("Build the network...")
        self.model_setup()
        print("Loss function calculations...")
        self.loss_calc()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("The log writer...")
            writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
            print("Initializing the global variables...")
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            print("Read input to nd array...")
            self.input_read(sess)
            print("Restore the model to run it from last checkpoint...")
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(ckpt_dir)
                if chkpt_fname is not None:
                    saver.restore(sess, chkpt_fname)
            print("Training Loop...")
            for epoch in range(0, max_epoch):
                print("In the epoch ", epoch)
                curr_lr = 2e-4 - epoch * 1e-5
                if (save_training_images):
                    print("Save the training images...")
                    self.save_training_images(sess, epoch)
                for ptr in range(0, max_images):
                    print("In the iteration ", ptr)

                    summary_str = None

                    # Optimizing the D_B network
                    for i in range(n_critic):
                        iter = (ptr + i) if (ptr + i) < 1000 else (ptr + i) - 1000
                        fake_B = sess.run(self.fake_B, feed_dict={self.input_A: self.A_input[iter]})
                        fake_B_temp = self.fake_image_pool(self.num_fake_inputs, fake_B, self.fake_images_B)
                        _, summary_str = sess.run(
                            [self.d_B_trainer, self.d_B_loss_summ],
                            feed_dict={
                                self.input_A: self.A_input[iter],
                                self.input_B: self.B_input[iter],
                                self.lr: curr_lr,
                                self.fake_pool_B: fake_B_temp}
                        )
                    writer.add_summary(summary_str, epoch * max_images + ptr)

                    # Optimizing the G_A network
                    _, summary_str = sess.run(
                        [self.g_A_trainer, self.g_A_loss_summ],
                        feed_dict={
                            self.input_A: self.A_input[ptr],
                            self.input_B: self.B_input[ptr],
                            self.lr: curr_lr}
                    )
                    writer.add_summary(summary_str, epoch * max_images + ptr)

                    # Optimizing the D_A network
                    for i in range(n_critic):
                        iter = (ptr + i) if (ptr + i) < 1000 else (ptr + i) - 1000
                        fake_A = sess.run(self.fake_A, feed_dict={self.input_B: self.B_input[iter]})
                        fake_A_temp = self.fake_image_pool(self.num_fake_inputs, fake_A, self.fake_images_A)
                        _, summary_str = sess.run(
                            [self.d_A_trainer, self.d_A_loss_summ],
                            feed_dict={
                                self.input_A: self.A_input[iter],
                                self.input_B: self.B_input[iter],
                                self.lr: curr_lr,
                                self.fake_pool_A: fake_A_temp}
                        )
                    writer.add_summary(summary_str, epoch * max_images + ptr)

                    # Optimizing the G_B network
                    _, summary_str = sess.run(
                        [self.g_B_trainer, self.g_B_loss_summ],
                        feed_dict={
                            self.input_A: self.A_input[ptr],
                            self.input_B: self.B_input[ptr],
                            self.lr: curr_lr}
                    )
                    writer.add_summary(summary_str, epoch * max_images + ptr)

                    self.num_fake_inputs += 1
                print("Save the model...")
                saver.save(sess, os.path.join(ckpt_dir, "drugan"), global_step=epoch)

    def test(self):

        ''' Testing Function'''

        self.input_setup()
        self.model_setup()
        saver = tf.train.Saver()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init)
            self.input_read(sess)
            chkpt_fname = tf.train.latest_checkpoint(ckpt_dir)
            print("Restore the model...")
            saver.restore(sess, chkpt_fname)
            if not os.path.exists("./output/test/"):
                os.makedirs("./output/test/")
            print("Testing loop...")
            for i in range(0, 1000):
                print("In the iteration ", i)
                fake_A_temp, fake_B_temp = sess.run(
                    [self.fake_A, self.fake_B],
                    feed_dict={
                        self.input_A: self.A_input[i],
                        self.input_B: self.B_input[i]}
                )
                imsave("./output/test/fakeA_" + str(i) + ".jpg", ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/test/fakeB_" + str(i) + ".jpg", ((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/test/inputA_" + str(i) + ".jpg", ((self.A_input[i][0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/test/inputB_" + str(i) + ".jpg", ((self.B_input[i][0] + 1) * 127.5).astype(np.uint8))


def main():
    model = DRUGAN()
    if to_train:
        model.train()
    # if to_test:
    #     model.test()


if __name__ == '__main__':
    main()
