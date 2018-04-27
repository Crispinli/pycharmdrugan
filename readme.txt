算法描述：
    （1）模型整体结构：
        a. 整体结构类似 CycleGAN/DualGAN/DiscoGAN 模型，并且进行了改进
        b. 模型中的包含两个 GAN 模型，并同时进行优化
        c. 两个 GAN 当中的生成器 generator 和判别器 discriminator 的结构相同
        d. 对每个 GAN 的生成器进行 1 次优化，然后对判别器进行 1 次优化
        e. 模型参数使用 xavier initializer 进行初始化
    （2）生成器 generator 的结构：
        a. 整体结构类似 U-Net 模型的形式
        b. 在 encoder 部分，编码结果直接与 decoder 部分的对应结果进行拼接
    （3）判别器 discriminator 结构：
        a. 整体结构为全卷积网络 FCN 的形式
        b. 输出是一个经过编码操作的 tensor，尺寸为 [4, 8, 8, 1]
        c. 输入是图像 patch 的形式，尺寸为 [4, 64, 64, 3]
    （4）模型的损失函数：
        a. 两个 GAN 的损失函数具有相同的形式
        b. 损失函数类似 WGAN_GP 的形式，并且进行了改进
        c. 判别器损失的计算方式不变，在生成器损失中加入重新定义的 cycle loss 项
    （5）模型训练策略：
        a. 最优化算法采用 tf.train.AdamOptimizer 算法
        b. 一次训练会进行 100 个 epoch，每个 epoch 中进行 1000 次迭代
        c. 初始学习率为 2e-4，每进行 1 个 epoch 的训练，学习率衰减 2e-6
        d. 训练中每个 epoch 都会打乱输入 A 和输入 B 的对应顺序
        e. 采用 Group Normalization 进行标准化