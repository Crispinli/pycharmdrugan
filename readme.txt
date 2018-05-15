算法描述：
    （1）模型整体结构：
        a. 整体结构类似 CycleGAN/DualGAN/DiscoGAN 模型，并且进行了改进
        b. 模型中的包含两个 GAN 模型，并同时进行优化
        c. 两个 GAN 当中的生成器 generator 和判别器 discriminator 的结构相同
        d. 对每个 GAN 的生成器进行 1 次优化，然后对判别器进行 5 次优化
        e. 模型参数使用 xavier initializer 进行初始化
    （2）生成器 generator 的结构：
        a. 整体结构为 multi-scale 的结构形式
        b. 多个尺度的图像块宽度分别为 256 128 64 32 16
        c. 较小图像块的卷积结果先经过双线性插值进行上采样后再与较大图像块的卷积结果在通道维度上进行拼接
        d. 生成器的初始通道数为 32，然后按照公比为 2 的形式逐层增加
    （3）判别器 discriminator 结构：
        a. 整体结构为 multi-scale 的形式
        b. 多个尺度的图像块宽度分别为 256 128 64
        c. 输出是一个经过编码操作的 tensor，尺寸为 [3, 8, 8, 1]
        d. 判别器的初始通道数为 32，然后按照公比为 2 的形式逐层增加
    （4）模型的损失函数：
        a. 两个 GAN 的损失函数具有相同的形式
        b. 损失函数类似 LSGAN 的形式
        c. 判别器损失的计算方式不变，在生成器损失中加入 reconstruction loss 项
    （5）模型训练策略：
        a. 最优化算法采用 tf.train.AdamOptimizer 算法
        b. 一次训练会进行 100 个 epoch，每个 epoch 中进行 1000 次迭代
        c. 初始学习率为 2e-4，每进行 1 个 epoch 的训练，学习率衰减 2e-6
        d. 训练中每个 epoch 都会打乱输入 A 和输入 B 的对应顺序
        e. 采用 Group Normalization 进行标准化