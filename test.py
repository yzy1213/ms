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
batch_size = 350

# LSTM网络的层数
num_layers = 2

# LSTM隐藏层的大小（神经元数量）
hidden_size = 220

# 输入特征的维度大小
input_size = 43

# 时间序列的长度（每个序列的时间步数）
sequence_length = 1

# 训练迭代的次数
epochs = 2000

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

    """

    def __init__(self, input, label):
        self._data = input
        self._label = label
    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)


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
        """
    def __init__(self, input_size, hidden_size, num_layers, batch_size):  # 子类构造函数有三个参数，input_size,hidden_size,num_layer
        super(Network, self).__init__()  # 子类构造函数调用父类构造函数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, has_bias=True, batch_first=False)
        self.out = nn.Dense(hidden_size, 1)

    def construct(self, x):
        a = self.num_layers
        b = self.batch_size
        c = self.hidden_size
        h = Tensor(np.zeros([a, b, c]), dtype=mindspore.float32)
        s = Tensor(np.zeros([a, b, c]), dtype=mindspore.float32)
        out, s_out = self.rnn(x, (h, s))  # 过LSTM网络
        output = self.out(out)  # 过线性层
        return output


# -----------------------------------------  读取测试集  ---------------------------------------------------------
data_path = 'cydata10_test_time.csv'

df = pd.read_csv(data_path)
all_train_Data = df.iloc[:, :].values
mylabel = all_train_Data[:, 0]  # 创建标签对象


new_df = df.copy()
for t in range(1, slide_window_size + 1):
    new_df[f't{t}'] = pd.Series(mylabel).shift(t)

new_df = new_df.drop(range(slide_window_size))
all_train_Data = new_df.iloc[:, :].values

scaler = MinMaxScaler()
all_Data_normalized = scaler.fit_transform(all_train_Data)

# 从归一化的数据中获取特征和标签
test_x = all_Data_normalized[:, 1:]
test_y = all_Data_normalized[:, 0]
# 将数据类型转换为Float32
test_x = test_x.astype(np.float32)
test_y = test_y.astype(np.float32)

# 变换为列向量
test_y = np.reshape(test_y, (-1, 1))

# 数据集迭代器定义
test_loader = ds.GeneratorDataset(source=MyAccessible(test_x, test_y), column_names=["data", "label"], shuffle=False)
test_loader = test_loader.batch(batch_size=batch_size, drop_remainder=True)

# 测试迭代器
# i = 0
# for inputs, labels in test_loader:
#     i = i + 1
#     print("----------------------labels--------------------------\n ", labels)
#
# print("i=", i)
input_size = input_size + slide_window_size
# 模型实例化
model = Network(input_size, hidden_size, num_layers, batch_size)

# 加载模型
model = Network(input_size, hidden_size, num_layers, batch_size)
param_dict = mindspore.load_checkpoint("model.ckpt")  # 请选择模型文件
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)


# -----------------------------------------  推理过程  ---------------------------------------------------------
model.set_train(False)  # 将模型设置为推理模式，不进行训练
for data, label in test_loader:
    data = data.reshape(sequence_length, batch_size, -1)  # 调整数据维度以适应模型输入
    pred = model(data)  # 对输入数据进行推理，得到预测结果
    print(f'Predicted: "{pred[:100]}", Actual: "{label[:100]}"')  # 打印部分预测结果和实际标签
    break

pred_numpy = pred[:].asnumpy().reshape(-1)  # 将预测结果转换为NumPy数组并重新调整形状
label_numpy = label[:].asnumpy().reshape(-1)  # 将实际标签转换为NumPy数组并重新调整形状


# -----------------------------------------  数据可视化  ---------------------------------------------------------
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
