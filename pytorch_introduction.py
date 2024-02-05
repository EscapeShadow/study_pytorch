import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as Data
from sklearn.datasets import load_iris

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def _2_2_1():  # 张量的数据类型
    # 获取张量的数据类型
    print(torch.tensor([1.2, 3.4]).dtype)

    # 将张量的默认数据类型设置为其他类型
    torch.set_default_dtype(torch.float64)
    print(torch.tensor([1.2, 3.4]).dtype)

    # 将张量数据类型转化为整型
    a = torch.tensor([1.2, 3.4])
    print("a.dtype", a.dtype)
    print("a.long()方法: ", a.long().dtype)
    print("a.int()方法:", a.int().dtype)
    print("a.float()方法:", a.float().dtype)

    # 恢复torch默认的数据类型
    torch.set_default_dtype(torch.float32)
    print(torch.tensor([1.2, 3.4]).dtype)

    # 获取torch默认的数据类型
    print(torch.get_default_dtype())


def _2_2_2_1():  # 张量的生成-torch.tensor()
    # python 的列表或序列可以通过torch.tensor()函数构造张量
    A = torch.tensor([[1.0, 1.0], [2, 2]])
    print(A)

    # 获取张量的维度
    print(A.shape)

    # 计算张量中包含的元素数量
    print(A.numel())

    # dtype 指定张量的数据类型/ requires_grad 指定张量是否需要计算梯度
    # 只有计算了梯度的张量，才能在深度网络优化时根据梯度大小进行更新
    B = torch.tensor((1, 2, 3), dtype=torch.float32, requires_grad=True)
    print(B)

    # 因为张量B是可计算梯度的,故可以计算sum(B**2)的梯度
    y = B.pow(2).sum()
    y.backward()
    print(B.grad)
    # 从输出结果上看每个位置上的梯度是2×B
    # 注意!!!: 只有浮点类型的张量才允许计算精度


def _2_2_2_2():  # 张量的生成-torch.Tensor()
    # 根据已有的数据创建张量
    C = torch.Tensor([1, 2, 3, 4])
    print(C)
    # 创建具有特定大小的张量
    D = torch.Tensor(2, 3)
    print(D)
    # 创建一个与D相同大小和类型的全1张量
    print(torch.ones_like(D))
    # 创建一个与D维度相同的全0张量
    print(torch.zeros_like(D))
    # 创建一个与D维度相同的随机张量
    print(torch.rand_like(D))
    # 创建一个类型相似但是尺寸不同的张量
    E = [[1, 2], [3, 4]]
    E = D.new_tensor(E)
    print("D.dtype :", D.dtype)
    print("E.dtype :", E.dtype)


def _2_2_2_3():  # 张量的生成-Numpy数据转换
    # 利用numpy数组生成张量
    F = np.ones((3, 3))
    # 使用torch.as_tensor()函数
    Ftensor = torch.as_tensor(F)
    print(Ftensor)

    # 使用torch.from_numpy()函数
    Ftensor = torch.from_numpy(F)
    print(Ftensor)
    # !! numpy生成的数组默认就是64位浮点型数组

    # 通过指定均值和标准差生成随机数
    torch.manual_seed(123)  # 指定生成随机数的种子，保证生成的随机数是可重复出现的
    A = torch.normal(mean=0.0, std=torch.arange(1, 5.0))
    print(A)

    # 可以分别指定每个随机数服从的均值
    print(torch.normal(mean=torch.arange(1, 5.0), std=torch.arange(1, 5.0)))

    # 在区间[0, 1]上生成服从均匀分布的张量
    torch.manual_seed(123)
    B = torch.rand(3, 4)
    print(B)

    # 生成和其他张量尺寸相同的随机数张量
    torch.manual_seed(123)
    C = torch.ones(2, 3)
    D = torch.rand_like(C)
    print(D)

    # 生成服从标准正态分布的随机张量
    print(torch.randn(3, 3))
    print(torch.randn_like(C))

    # 将0-10(不包含10)之间的整数随机排序
    torch.manual_seed(123)
    torch.randperm(10)

    # 使用torch.arange()生成张量
    print(torch.arange(start=0, end=10, step=2))

    # 在范围内生成固定数量的等间隔张量
    print(torch.linspace(start=0, end=10, steps=5))

    # 生成以对数为间隔的张量
    print(torch.logspace(start=0.1, end=1.0, steps=5))


def _2_2_3_1():  # 张量操作-改变张量的形状
    # 使用tensor.reshape()方法设置张量的形状大小
    A = torch.arange(12.0).reshape(3, 4)
    print(A)

    # 使用torch.reshape()
    print(torch.reshape(input=A, shape=(2, -1)))

    # 使用resize()方法
    print(A.resize_(2, 6))

    # 使用resize_as_()方法复制张量的形状大小
    B = torch.arange(10.0, 19.0).reshape(3, 3)
    print(B)
    print(A.resize_as_(B))

    # torch.unsqueeze()函数在指定维度插入尺寸为1的新张量
    A = torch.arange(12.0).reshape(2, 6)
    B = torch.unsqueeze(A, dim=0)
    print(B.shape)

    # torch.squeeze()函数移除所有维度为1的维度
    C = B.unsqueeze(dim=3)
    print("C.shape :", C.shape)
    D = torch.squeeze(C)
    print("D.shape :", D.shape)

    # 移除指定维度为1的维度
    E = torch.squeeze(C, dim=0)
    print("E.shape :", E.shape)

    # 使用.expand()方法扩展张量
    A = torch.arange(3)
    B = A.expand(3, -1)
    print(B)
    # 使用.expand_as()方法扩展张量
    C = torch.arange(6).reshape(2, 3)
    B = A.expand_as(C)
    print(B)

    # 使用.repeat()方法扩展张量
    D = B.repeat(1, 2, 2)
    print(D)
    print(D.shape)


def _2_2_3_2():  # 获张量操作-取张量中的元素
    # 利用切片和索引获取张量中的元素
    A = torch.arange(12).reshape(1, 3, 4)
    print(A)
    print(A[0])
    # 获取第0维度下的矩阵前两行元素
    print(A[0, 0:2, :])
    # 获取第0维度下的矩阵,最后一行-4~-1列
    print(A[0, -1, -4:-1])

    # 根据条件筛选
    B = -A
    print(torch.where(A > 5, A, B))
    # 获取A中大于5的元素
    print(A[A > 5])

    # 获取矩阵张量下三角部分
    print(torch.tril(A, diagonal=0))
    # diagonal参数控制要考虑的对角线
    print(torch.tril(A, diagonal=1))
    # 获取矩阵张量的上三角部分
    print(torch.triu(A, diagonal=0))
    # 获取矩阵张量的对角线元素,input 需要是一个二维的张量
    C = A.reshape(3, 4)
    print(C)
    print(torch.diag(C, diagonal=0))
    print(torch.diag(C, diagonal=1))  # diagonal是相对于对角线的位移

    # 提供对角线元素生成矩阵张量
    print(torch.diag(torch.tensor([1, 2, 3])))


def _2_2_3_3():  # 张量操作-拼接和拆分
    # 在给定维度中连接给定的张量序列
    A = torch.arange(6.0).reshape(2, 3)
    B = torch.linspace(0, 10, 6).reshape(2, 3)
    # 在0维度连接张量
    C = torch.cat((A, B), dim=0)
    print(B)
    print(C)
    # 在1维度连接张量
    D = torch.cat((A, B), dim=1)
    print(D)
    # 在1维度连接三个张量
    E = torch.cat((A[:, 1:2], A, B), dim=1)
    print(E)

    # 沿新维度连接张量
    F = torch.stack((A, B), dim=0)
    print(F)
    print(F.shape)
    G = torch.stack((A, B), dim=2)
    print(G)
    print(G.shape)

    # 在行上将张量E分为两块
    print(torch.chunk(E, 2, dim=0))
    D1, D2 = torch.chunk(D, 2, dim=1)
    print(D1)
    print(D2)
    # 如果沿着给定维度dim的张量大小不能被块整除，则最后一个块将最小
    E1, E2, E3 = torch.chunk(E, 3, dim=1)
    print(E1)
    print(E2)
    print(E3)
    # 将张量切分为块, 指定每个块的大小
    D1, D2, D3 = torch.split(D, [1, 2, 3], dim=1)
    print(D1)
    print(D2)
    print(D3)


def _2_2_4_1():  # 张量计算-比较大小
    # 比较两个数是否接近
    A = torch.tensor([10.0])
    B = torch.tensor([10.1])
    print(torch.allclose(A, B, rtol=1e-05, atol=1e-08, equal_nan=False))
    print(torch.allclose(A, B, rtol=0.1, atol=0.01, equal_nan=False))  # 不同的比较标准会有不同的判断结果

    # 如果equal_nan=True, 那么缺失值可以判断接近
    A = torch.tensor(float("nan"))
    print(torch.allclose(A, A, equal_nan=False))
    print(torch.allclose(A, A, equal_nan=True))

    # 计算元素是否相等
    A = torch.tensor([1, 2, 3, 4, 5, 6])
    B = torch.arange(1, 7)
    C = torch.unsqueeze(B, dim=0)
    print(B)
    print(C)
    print(torch.eq(A, B))
    print(torch.eq(A, C))
    # 判断两个张量是否具有相同的形状和元素
    print(torch.equal(A, B))
    print(torch.equal(A, C))

    # 逐元素比较大于等于
    print(torch.ge(A, B))
    print(torch.ge(A, C))
    # 逐元素比较大于
    print(torch.gt(A, B))
    print(torch.gt(A, C))

    # 逐元素比较小于等于
    print(torch.le(A, B))  # 小于等于
    print(torch.lt(A, C))  # 小于

    # 逐元素比较不等于
    print(torch.ne(A, B))
    print(torch.ne(A, C))
    # 判断是否为缺失值
    print(torch.isnan(torch.tensor([0, 1, float("nan"), 2])))


def _2_2_4_2():  # 张量计算-基本运算
    # 矩阵逐元素相乘
    A = torch.arange(6.0).reshape(2, 3)
    B = torch.linspace(10, 20, steps=6).reshape(2, 3)
    print("A:", A)
    print("B:", B)
    print("A * B:", A * B)
    # 逐元素相除
    print(A / B)
    # 逐元素相加
    print(A + B)
    # 逐元素相减
    print(A - B)
    # 逐元素整除
    print(B // A)

    # 张量的幂
    print(torch.pow(A, 3))
    print(A ** 3)

    # 计算张量的指数
    print(torch.exp(A))
    # 计算张量的对数
    print(torch.log(A))
    # 计算张量的平方根
    print(torch.sqrt(A))
    print(A ** 0.5)
    # 计算张量的平方根倒数
    print(torch.rsqrt(A))
    print(1 / (A ** 0.5))

    # 根据最大值裁剪
    print(torch.clamp_max(A, 4))
    # 根据最小值裁剪
    print(torch.clamp_min(A, 3))
    # 根据范围裁剪
    print(torch.clamp(A, 2.5, 4))

    # 矩阵的转置
    C = torch.t(A)
    print(C)
    # 矩阵运算，矩阵相乘，A的行数要等于C的列数
    print(A.matmul(C))
    A = torch.arange(12.0).reshape(2, 2, 3)
    B = torch.arange(12.0).reshape(2, 3, 2)
    AB = torch.matmul(A, B)
    print(A)
    print(B)
    print(AB)
    # 矩阵相乘只计算最后面的两个维度的乘法
    print(AB[0].eq(torch.matmul(A[0], B[0])))
    print(AB[1].eq(torch.matmul(A[1], B[1])))

    # 计算矩阵的逆
    C = torch.rand(3, 3)
    D = torch.inverse(C)
    print(C)
    print(torch.mm(C, D))
    # 计算张量矩阵的迹, 对角线元素的和
    print(torch.trace(torch.arange(9.0).reshape(3, 3)))


def _2_2_4_3():  # 张量计算-统计相关的计算
    # 一维张量的最大值和最小值
    A = torch.tensor([12., 34, 25, 11, 67, 32, 29, 30, 99, 55, 23, 44])
    # 最大值及位置
    print("最大值：", A.max())
    print("最大值位置: ", A.argmax())
    # 最小值及位置
    print("最小值:", A.min())
    print("最小值位置：", A.argmin())

    # 二维张量的最大值和最小值
    B = A.reshape(3, 4)
    print("2-D张量B:\n", B)
    # 最大值及位置(每行)
    print("最大值: \n", B.max(dim=1))
    print("最大值位置:\n", B.argmax(dim=1))
    # 最小值及位置(每列)
    print("最小值: \n", B.min(dim=0))
    print("最小值: \n", B.argmin(dim=0))

    # 张量排序,分别输出从小到大的排序结果和相应的元素在原始位置的索引
    print(torch.sort(A))
    # 按照降序进行排列
    print(torch.sort(A, descending=True))
    # 对2-D张量进行排序
    Bsort, Bsort_id = torch.sort(B)
    print("B sort:\n", Bsort)
    print("B sort_id:\n", Bsort_id)
    print("B argsort:\n", torch.argsort(B))

    # 获取张量前几个大的数值
    print(torch.topk(A, 4))

    # 获取2-D张量每列前几个大的数值
    Btop2, Btop2_id = torch.topk(B, 2, dim=0)
    print("B 每列 top2:\n", Btop2)
    print("B 每列 top2 位置:\n", Btop2_id)
    # 获取张量第k小的数值和位置
    print(torch.kthvalue(A, 3))
    # 获取2-D张量第k小的数值和位置
    print(torch.kthvalue(B, 3, dim=1))
    # 获取2-D张量第k小的数值和位置
    Bkth, Bkth_id = torch.kthvalue(B, 3, dim=1, keepdim=True)
    print("Bkth:", Bkth)

    # 平均值, 计算每行的均值
    print(torch.mean(B, dim=1, keepdim=True))
    # 平均值, 计算每列的均值
    print(torch.mean(B, dim=0, keepdim=True))
    # 计算每行的和
    print(torch.sum(B, dim=1, keepdim=True))
    # 计算每列的和
    print(torch.sum(B, dim=0, keepdim=False))
    # 按照行计算累加和
    print(torch.cumsum(B, dim=1))
    # 按照列计算累加和
    print(torch.cumsum(B, dim=0))
    # 计算每行的中位数
    print(torch.median(B, dim=1, keepdim=True))
    # 计算每列的中位数
    print(torch.median(B, dim=0, keepdim=True))
    # 按照行计算乘积
    print(torch.prod(B, dim=1, keepdim=True))
    # 按照列计算乘积
    print(torch.prod(B, dim=0, keepdim=True))
    # 按照行计算累乘积
    print(torch.cumprod(B, dim=1))
    # 按照列计算累乘积
    print(torch.cumprod(B, dim=0))
    # 标准差
    print(torch.std(A))


def _2_3():  # PyTorch中的自动微分
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    # 默认requires_grad=False
    y = torch.sum(x ** 2 + 2 * x + 1)
    print("x.requires_grad:", x.requires_grad)
    print("y.requires_grad:", y.requires_grad)
    print("x:", x)
    print("y:", y)

    # 计算y在x上的梯度
    y.backward()  # 自动计算出每个元素上的导数值
    print(x.grad)  # 通过grad属性获取此时x的梯度信息


def _2_4_1():  # torch.nn模块-卷积层
    # 使用一张图像来展示经过卷积后的图像效果

    # 读取图像->转化为灰度图片->转化为numpy数组
    myim = Image.open("source/yellow.jpg")
    myimgray = np.array(myim.convert("L"), dtype=np.float32)
    # 可视化图片
    # plt.figure(figsize=(6, 6))
    # plt.imshow(myimgray, cmap="gray")
    # plt.axis("off")
    # plt.show()

    # 将数组转化为张量
    imh, imw = myimgray.shape
    myimgray_t = torch.from_numpy(myimgray.reshape((1, 1, imh, imw)))
    print(myimgray_t.shape)

    # 对灰度图像进行卷积提取图像轮廓
    kersize = 5  # 定义边缘检测卷积核, 并将维度处理为1*1*5*5
    ker = torch.ones(kersize, kersize, dtype=torch.float32) * (-1)
    ker[2, 2] = 24
    ker = ker.reshape((1, 1, kersize, kersize))
    # 进行卷积操作
    conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)
    # 设置卷积时使用的核,第一个核使用边缘检测核
    conv2d.weight.data[0] = ker
    # 对灰度图像进行卷积操作
    imconv2dout = conv2d(myimgray_t)
    # 对卷积后的输出进行维度压缩
    imconv2dout_im = imconv2dout.data.squeeze()
    print("卷积后的尺寸:", imconv2dout_im.shape)
    # 可视化卷积后的图像
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(imconv2dout_im[0], cmap="gray")
    # plt.axis("off")
    # plt.subplot(1, 2, 2)
    # plt.imshow(imconv2dout_im[1], cmap="gray")
    # plt.axis("off")
    # plt.show()
    # 使用边缘特征提取卷积核很好的提取出了图像的边缘信息
    # 右边的图像使用的卷积核为随机数, 得到的卷积结果与原始图像很相似

    return imconv2dout


def _2_4_2():  # torch.nn模块-池化层
    # 对卷积后的结果进行最大值池化
    maxpool2 = nn.MaxPool2d(2, stride=2)
    pool2_out = maxpool2(_2_4_1())
    pool2_out_im = pool2_out.squeeze()
    print(pool2_out.shape)

    # 可视化最大值池化后的结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pool2_out_im[0].data, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pool2_out_im[1].data, cmap="gray")
    plt.axis("off")
    plt.show()

    # 对卷积后的结果进行平均值池化
    avgpool2 = nn.AvgPool2d(2, stride=2)
    pool2_out = avgpool2(_2_4_1())
    pool2_out_im = pool2_out.squeeze()
    print(pool2_out.shape)
    # 可视化平均值池化后的结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pool2_out_im[0].data, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pool2_out_im[1].data, cmap="gray")
    plt.axis("off")
    plt.show()

    # 对卷积后的结果进行自适应平均值池化
    AdaAvgpool2 = nn.AdaptiveAvgPool2d(output_size=(100, 100))
    pool2_out = AdaAvgpool2(_2_4_1())
    pool2_out_im = pool2_out.squeeze()
    print(pool2_out.shape)
    # 可视化自适应平均值池化后的结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pool2_out_im[0].data, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pool2_out_im[1].data, cmap="gray")
    plt.axis("off")
    plt.show()


def _2_4_3():  # torch.nn模块-激活函数
    # 可视化几种激活函数的图像
    x = torch.linspace(-6, 6, 100)
    sigmoid = nn.Sigmoid()  # Sigmoid激活函数
    ysigmoid = sigmoid(x)
    tanh = nn.Tanh()  # Tanh激活函数
    ytanh = tanh(x)
    relu = nn.ReLU()  # ReLU激活函数
    yrelu = relu(x)
    softplus = nn.Softplus()  # Softplus激活函数
    ysoftplus = softplus(x)

    plt.figure(figsize=(14, 3))  # 可视化激活函数
    plt.subplot(1, 4, 1)
    plt.plot(x.data.numpy(), ysigmoid.data.numpy(), "r-")
    plt.title("Sigmoid")
    plt.grid()
    plt.subplot(1, 4, 2)
    plt.plot(x.data.numpy(), ytanh.data.numpy(), "r-")
    plt.title("Tanh")
    plt.grid()
    plt.subplot(1, 4, 3)
    plt.plot(x.data.numpy(), yrelu.data.numpy(), "r-")
    plt.title("ReLU")
    plt.grid()
    plt.subplot(1, 4, 4)
    plt.plot(x.data.numpy(), ysoftplus.data.numpy(), "r-")
    plt.title("Softplus")
    plt.grid()
    plt.show()


def _2_5_1():  # Pytorch预处理-高维数组
    # 回归数据准备-读取波士顿回归数据
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    # housing = fetch_california_housing()
    print(data.dtype, target.dtype)

    # 训练集X转化为张量, 训练集y转化为张量
    train_xt = torch.from_numpy(data.astype(np.float32))
    train_yt = torch.from_numpy(data.astype(np.float32))
    print(train_xt.dtype, train_yt.dtype)

    # 将训练集转化为张量后, 使用TensorDataset将X和Y整理到一起
    train_data = Data.TensorDataset(train_xt, train_yt)
    # 定义一个数据加载器, 将训练数据集进行批量处理
    train_loader = Data.DataLoader(
        dataset=train_data,  # 使用的数据集
        batch_size=64,  # 批处理样本大小
        shuffle=True,  # 每次迭代前打乱数据
        num_workers=2  # 使用两个进程
    )

    # 检查训练数据集的一个batch样本的维度是否正确
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break

    # 输出训练图像的尺寸和标签的尺寸及数据类型
    print("b_x.shape:", b_x.shape)
    print("b_y.shape:", b_y.shape)
    print("b_x.dtype:", b_x.dtype)
    print("b_y.dtype:", b_y.dtype)

    # 分类数据准备-处理分类数据
    iris_x, iris_y = load_iris(return_X_y=True)
    print("iris_x.dtype:", iris_x.dtype)
    print("iris_y.dtype:", iris_y.dtype)

    # 训练集x转化为张量, 训练集y转化为张量
    train_xt = torch.from_numpy(iris_x.astype(np.float32))
    train_yt = torch.from_numpy(iris_y.astype(np.int64))
    print(train_xt.dtype, train_yt.dtype)

    # 将训练集转化为张量后, 使用TensorDataset将X和Y整理到一起
    train_data = Data.TensorDataset(train_xt, train_yt)
    # 定义一个数据加载器, 将训练数据集进行批量处理
    train_loader = Data.DataLoader(
        dataset=train_data,  # 使用的数据集
        batch_size=10,  # 批处理样本大小
        shuffle=True,  # 每次迭代前打乱数据
        num_workers=1  # 使用两个进程
    )

    # 检查训练数据集的一个batch样本的维度是否正确
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break

    # 输出训练图像的尺寸和标签的尺寸及数据类型
    print("b_x.shape:", b_x.shape)
    print("b_y.shape:", b_y.shape)
    print("b_x.dtype:", b_x.dtype)
    print("b_y.dtype:", b_y.dtype)


def _2_5_2():  # pytorch预处理-图像数据
    # 从torchvision中的datasets模块中导入数据并预处理
    # 使用FashionMNIST数据, 准备训练数据集
    train_data = FashionMNIST(
        root="./data/FashionMNIST",  # 数据的路径
        train=True,  # 只使用训练数据集
        transform=transforms.ToTensor(),
        download=False,
    )
    # 定义一个数据加载器
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    # 计算train_loader有多少个batch
    print("train_loader的batch数量为:", len(train_loader))

    # 对测试集进行处理
    test_data = FashionMNIST(
        root="./data/FashionMNIST",
        train=False,
        download=False
    )
    # 为数据添加一个通道维度, 并且取值范围缩放到0~1之间
    test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
    test_data_x = torch.unsqueeze(test_data_x, dim=1)
    test_data_y = test_data.targets
    print(test_data_x.shape, test_data_y.shape)


if __name__ == '__main__':
    # _2_2_1()  # 张量的数据类型
    # _2_2_2_1()  # 张量的生成-torch.tensor()
    # _2_2_2_2()  # 张量的生成-torch.Tensor()
    # _2_2_2_3()  # 张量的生成-Numpy数据转换
    # _2_2_3_1()  # 张量操作-改变张量的形状
    # _2_2_3_2()  # 张量操作-获取张量中的元素
    # _2_2_3_3()  # 张量操作-拼接和拆分
    # _2_2_4_1()  # 张量计算-比较大小
    # _2_2_4_2()  # 张量计算-基本运算
    # _2_2_4_3()  # 张量计算-统计相关的计算
    # _2_3()  # PyTorch中的自动微分
    # _2_4_1()  # torch.nn模块-卷积层
    # _2_4_2()  # torch.nn模块-池化层
    # _2_4_3()  # torch.nn模块-激活函数
    # _2_5_1()  # Pytorch预处理-高维数组
    _2_5_2()  # pytorch预处理-图像数据
