# 本程序修改了sequence_lenth可以修改
import csv
import numpy as np
from mindspore import nn
from mindspore import Tensor
import mindspore
import mindspore.dataset as ds
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# -------------------- 参数初始化 --------------------------
# 每个批次中的样本数量
batch_size = 1024

# 学习率（用于优化器的学习速率）
lr = 0.0001

# 权重衰减（用于控制正则化项的大小，有助于防止过拟合）
weight_decay = 0.00001

# LSTM网络的层数
num_layers = 2

# LSTM隐藏层的大小（神经元数量）
hidden_size = 220

# 输入特征的维度大小
input_size = 43

# 时间序列的长度（每个序列的时间步数）
sequence_length = 1

# 训练迭代的次数
epochs = 3000

# 时间滑窗大小
slide_window_size = 4

class MyAccessible:
    """
    自定义数据容器类，用于封装输入特征和标签，以便在数据加载过程中使用。

    Attributes:
        _data (list or numpy.ndarray): 输入特征数据。
        _label (list or numpy.ndarray): 标签数据。

    Methods:
        __init__(self, input, label): 类的构造函数，初始化输入特征和标签数据。
        __getitem__(self, index): 获取指定索引处的数据对，即输入特征和对应的标签。
        __len__(self): 返回数据容器中的数据数量。

    Example:
        input_data = [array([0.1, 0.2]), array([0.3, 0.4])]
        label_data = [0.5, 0.6]
        data_container = MyAccessible(input_data, label_data)
        for input_sample, label_sample in data_container:
            # 在这里使用 input_sample 和 label_sample 进行处理
    """

    def __init__(self, input, label):
        """
        构造函数，初始化数据容器。

        Args:
            input (list or numpy.ndarray): 输入特征数据。
            label (list or numpy.ndarray): 标签数据。
        """
        self._data = input
        self._label = label

    def __getitem__(self, index):
        # 调整以创建长度为 2 的序列
        start_idx = index
        end_idx = index + sequence_length
        return self._data[start_idx:end_idx], self._label[end_idx - 1]

    def __len__(self):
        """
        返回数据容器中的数据数量。

        Returns:
            int: 数据数量。
        """
        return len(self._data) - sequence_length


class Network(nn.Cell):
    """
    自定义网络模型类，基于nn.Cell，用于时间序列预测任务。

    Attributes:
        input_size (int): 输入特征的维度大小。
        hidden_size (int): 隐藏层的大小（神经元数量）。
        num_layers (int): LSTM网络的层数。
        batch_size (int): 每个批次中的样本数量。

    Methods:
        __init__(self, input_size, hidden_size, num_layers, batch_size):
            类的构造函数，初始化模型的各个参数。
        construct(self, x):
            构建模型的前向传播过程。

    Example:
        input_size = 43
        hidden_size = 80
        num_layers = 1
        batch_size = 512
        model = Network(input_size, hidden_size, num_layers, batch_size)
        output = model(input_data)
    """

    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        """
        构造函数，初始化模型的参数。

        Args:
            input_size (int): 输入特征的维度大小。
            hidden_size (int): 隐藏层的大小（神经元数量）。
            num_layers (int): LSTM网络的层数。
            batch_size (int): 每个批次中的样本数量。
        """
        super(Network, self).__init__()  # 子类构造函数调用父类构造函数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = mindspore.nn.LSTM(input_size, hidden_size, num_layers, has_bias=True, batch_first=False)
        self.out = nn.Dense(hidden_size, 1)

    def construct(self, x):
        a = self.num_layers
        b = self.batch_size
        c = self.hidden_size
        h = Tensor(np.zeros([a, b, c]), dtype=mindspore.float32)
        s = Tensor(np.zeros([a, b, c]), dtype=mindspore.float32)
        # 调整输入数据形状以匹配新的序列长度和特征数量
        x = x.view(sequence_length, -1, input_size)
        # print("x.shape", x.shape)
        out, s_out = self.rnn(x, (h, s))
        output = self.out(out)
        return output

# -----------------------------------------  读取训练集  ---------------------------------------------------------

data_path = 'cydata10_train_time.csv'

df = pd.read_csv(data_path)
all_train_Data = df.iloc[:, :].values

mylabel = all_train_Data[:, 0]

new_df = df.copy()
for t in range(1, slide_window_size + 1):
    new_df[f't{t}'] = pd.Series(mylabel).shift(t)

new_df = new_df.drop(range(slide_window_size))
input_size = input_size + slide_window_size
all_train_Data = new_df.iloc[:, :].values


# 对所有数据进行归一化处理
scaler = MinMaxScaler()
all_Data_normalized = scaler.fit_transform(all_train_Data)

# 从归一化的数据中获取特征和标签
train_x = all_Data_normalized[:, 1:]  # 提取除第一列外的所有特征列作为输入特征
train_y = all_Data_normalized[:, 0]   # 提取第一列作为标签

# 将数据类型转换为Float32
train_x = train_x.astype(np.float32)
train_y = train_y.astype(np.float32)

# 变换为列向量
train_y = np.reshape(train_y, (-1, 1))  # 将标签数据转换为列向量形式

# 创建训练集数据加载器
train_loader = ds.GeneratorDataset(source=MyAccessible(train_x, train_y), column_names=["data", "label"], shuffle=False)
train_loader = train_loader.batch(batch_size=batch_size, drop_remainder=True)


# 测试迭代器
# i = 0
# for inputs, labels in train_loader:
#     i = i + 1
#     inputs = inputs.view(sequence_length, -1, input_size)
#
#     print("输入数据：", inputs)
#     print("输入数据：", labels)
#     break
#     print("输入数据格式", inputs.shape)
#     print("标签数据格式", labels.shape)
# print("i=", i)

# -----------------------------------------  模型实例化  ---------------------------------------------------------
model = Network(input_size, hidden_size, num_layers, batch_size)  # 使用之前定义的 Network 类创建模型实例

loss_fn = mindspore.nn.MSELoss()  # 定义均方误差损失函数（Mean Squared Error）
optimizer = mindspore.nn.Adam(model.trainable_params(), learning_rate=lr, weight_decay=weight_decay)  # 定义优化器

def forward_fn(data, label, hidden_size, num_layers, batch_size):
    """
    前向传播函数，计算模型的输出和损失。

    Args:
        data (Tensor): 输入特征数据张量。
        label (Tensor): 实际标签数据张量。
        hidden_size (int): 隐藏层的大小（神经元数量）。
        num_layers (int): LSTM网络的层数。
        batch_size (int): 每个批次中的样本数量。

    Returns:
        tuple: 包含损失和模型输出的元组。
    """
    batch_size = data.shape[0]  # 更新批次大小
    data = data.reshape(sequence_length, batch_size, -1)  # 重新调整数据维度
    logits = model(data)  # 使用模型进行前向传播计算
    loss = loss_fn(logits, label)  # 计算损失
    return loss, logits


# 梯度函数
grad_fn = mindspore.value_and_grad(forward_fn, None, weights = model.trainable_params(), has_aux=True)

def train_step(data, label):
    """
    单个训练步骤函数，用于执行模型的前向传播、反向传播和参数更新。
    Args:
        data (Tensor): 输入特征数据张量。
        label (Tensor): 实际标签数据张量。
    Returns:
        float: 训练损失。
    """
    (loss, logits), grads = grad_fn(data, label, hidden_size, num_layers, batch_size)  # 执行前向传播和反向传播，计算梯度
    optimizer(grads)  # 使用优化器更新模型参数
    return loss  # 返回训练损失

def train(model, dataset):
    """
    模型训练函数，用于执行整个训练过程。
    Args:
        model (Network): 待训练的网络模型。
        dataset (GeneratorDataset): 训练数据加载器。
    Returns:
        None
    """
    loss_sum = 0
    model.set_train()  # 设置模型为训练模式
    for data, label in dataset:
        loss = train_step(data, label)  # 执行单个训练步骤，计算损失并更新参数
        loss_sum = loss_sum + loss
    print("loss_sum:", loss_sum)  # 打印本次训练的总损失

# -----------------------------------------  开始训练  ---------------------------------------------------------
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_loader)

mindspore.save_checkpoint(model, "model.ckpt")
print("---------- 模型保存完毕 ----------")


# -----------------------------------------  加载模型 ---------------------------------------------------------
model = Network(input_size, hidden_size, num_layers, batch_size)  # 创建一个与之前定义的 Network 类相同参数的模型实例
param_dict = mindspore.load_checkpoint("model.ckpt")  # 从文件中加载模型参数
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)  # 将加载的模型参数加载到创建的模型实例中


# -----------------------------------------  推理过程  ---------------------------------------------------------
# 这里使用训练集推理，使用测试集请运行test.py
model.set_train(False)  # 将模型设置为推理模式，不进行训练
for data, label in train_loader:
    data = data.reshape(sequence_length, batch_size, -1)  # 调整数据维度以适应模型输入
    pred = model(data)  # 对输入数据进行推理，得到预测结果
    # print(f'Predicted: "{pred[:100]}", Actual: "{label[:100]}"')  # 打印部分预测结果和实际标签
    break

pred_numpy = pred[:].asnumpy().reshape(-1)  # 将预测结果转换为NumPy数组并重新调整形状
label_numpy = label[:].asnumpy().reshape(-1)  # 将实际标签转换为NumPy数组并重新调整形状


# ----------------------------------------  数据可视化  --------------------------------------------------------
plt.figure(figsize=(10, 5))
# 绘制预测值和实际标签的对比
plt.plot(label_numpy[:], label='Actual', marker='o')
plt.plot(pred_numpy[:], label='Predicted', linestyle='dashed', marker='x')


# 添加图例
plt.legend()
# 添加标题和标签
plt.title('Predicted vs Actual Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label')
# 显示图像
plt.show()
