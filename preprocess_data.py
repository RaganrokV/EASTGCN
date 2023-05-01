import numpy as np
import torch
import torch.utils.data as Data


def divide_data(data, train_size, val_size, seq_len, pre_len):
    """
    划分数据为训练集、验证集和测试集

    参数：
    data: np.array，原始数据，shape为 (样本数量, 特征数量)
    train_size: int，训练集样本数量
    val_size: int，验证集样本数量
    seq_len: int，输入序列长度
    pre_len: int，预测序列长度

    返回：
    trainX: torch.Tensor，训练集输入数据，shape为 (训练样本数量, 输入序列长度, 特征数量)
    trainY: torch.Tensor，训练集预测数据，shape为 (训练样本数量, 预测序列长度, 特征数量)
    valX: torch.Tensor，验证集输入数据，shape为 (验证样本数量, 输入序列长度, 特征数量)
    valY: torch.Tensor，验证集预测数据，shape为 (验证样本数量, 预测序列长度, 特征数量)
    testX: torch.Tensor，测试集输入数据，shape为 (测试样本数量, 输入序列长度, 特征数量)
    testY: torch.Tensor，测试集预测数据，shape为 (测试样本数量, 预测序列长度, 特征数量)
    """
    train_data = data[:train_size, :]
    val_data = data[train_size:train_size+val_size, :]
    test_data = data[train_size+val_size:, :]

    trainX, trainY, valX, valY, testX, testY = [], [], [], [], [], []
    for i in range(np.size(train_data, 0) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len, :]
        trainX.append(a[0: seq_len, :])
        trainY.append(a[seq_len: seq_len + pre_len, :])
    for i in range(np.size(val_data, 0) - seq_len - pre_len):
        b = val_data[i: i + seq_len + pre_len, :]
        valX.append(b[0: seq_len, :])
        valY.append(b[seq_len: seq_len + pre_len, :])
    for i in range(np.size(test_data, 0) - seq_len - pre_len):
        c = test_data[i: i + seq_len + pre_len, :]
        testX.append(c[0: seq_len, :])
        testY.append(c[seq_len: seq_len + pre_len, :])

    trainX = torch.tensor(np.array(trainX))
    trainY = torch.tensor(np.array(trainY))
    valX = torch.tensor(np.array(valX))
    valY = torch.tensor(np.array(valY))
    testX = torch.tensor(np.array(testX))
    testY = torch.tensor(np.array(testY))

    return trainX, trainY, valX, valY, testX, testY



# def get_dataloader(Norm_TS, train_size, seq_len, pre_len, batch_size):
#     trainX, trainY, testX, testY = divide_data(data=Norm_TS, train_size=train_size, seq_len=seq_len,
#                                                pre_len=pre_len)
#     trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
#     testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()
#
#     train_dataset = Data.TensorDataset(trainX, trainY)
#     test_dataset = Data.TensorDataset(testX, testY)
#
#     # put into loader
#     train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#
#     return train_loader, test_loader, testX, testY
