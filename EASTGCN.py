# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import preprocessing
import scipy.sparse as sp

from torch_geometric_temporal.nn.recurrent.dcrnn import DConv
from torch.nn.utils import weight_norm
import torch
from torch import nn
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import  divide_data
import torch.utils.data as Data
import time

import torch.nn.functional as F

#%%
class ASTCG_net(torch.nn.Module):
    def __init__(self, in_channels, embed_channels, out_channels,adj_matrix):
        super(ASTCG_net, self).__init__()
        """:param"""
        self.ADJ=torch.tensor(adj_matrix).to(device)
        self.embed_size=12
        self.weight_inflation=25

        self.Spatial_cell = DConv(in_channels, embed_channels, 2)
        # self.Temporal_cell =nn.LSTM(in_channels,embed_channels,num_layers=2,dropout=0.5,
        #                             batch_first=True,bidirectional=True)
        self.Temporal_cell=TemporalConvNet(num_inputs=in_channels, num_channels=[64,embed_channels],
                                           kernel_size=2, dropout=0.1)

        """decoder part"""
        self.decoder_rnn=torch.nn.GRU(self.embed_size,
                          embed_channels//2,
                          num_layers=1,
                          batch_first=True)
        self.decoder_lin= nn.Linear(embed_channels//2, out_channels)

        self.Att=nn.MultiheadAttention(in_channels, 2, batch_first=True)

        self.linear=torch.nn.Linear(embed_channels*3, out_channels)
        # self.embedding = torch.nn.Linear(embed_channels * 3, self.embed_size)
        self.embedding = torch.nn.Linear(embed_channels +embed_channels, self.embed_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()



    def forward(self, x, edge_index, edge_weight ):

        """att weight"""
        _, attn_output_weights = self.Att(x, x, x)

        Mean_weight = torch.mean(self.ADJ * attn_output_weights, dim=0)
        index = torch.nonzero(Mean_weight, as_tuple=True)
        attention_scorce = Mean_weight[index[0], index[1]]
        Att_weigth=edge_weight*attention_scorce


        Spatial_info = self.Spatial_cell(x, edge_index, self.weight_inflation*Att_weigth) #Spatial_info(B, N_nodes, F_out)

        Spatial_info = F.relu(Spatial_info)

        Temporal_info = self.Temporal_cell(x.permute(0,2,1))
        # Temporal_info , _ = self.Temporal_cell(x)
        Temporal_info = F.relu(Temporal_info.permute(0,2,1))


        CAT = torch.cat((Temporal_info, Spatial_info), dim=2)

        embeding=self.embedding(CAT)

        h_feature, _=self.decoder_rnn(embeding)
        h_feature = F.relu(h_feature)
        Fusion=self.decoder_lin(h_feature)

        return Fusion

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs=24, n_outputs=1, kernel_size=2, stride=1, dilation=None, padding=3, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        if dilation is None:
            dilation = 2
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

#%%  hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len=24
pre_len=1
batch_size=32
train_size=3500 #80%
# set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#%%
data_csv = pd.read_csv(r'3-Unfixed sampling/periodic_sample.csv')
adj=pd.read_csv(r'3-Unfixed sampling//ADJ.csv')
adj_matrix=adj
# data_csv = pd.read_csv(r'periodic_sample.csv') #for debug
# adj=pd.read_csv(r'ADJ.csv')
# adj_matrix=adj
# data_csv = pd.read_csv(r'C:\Users\admin\Desktop\My_master_piece_LOL\data\SZ\sz_speed.csv')
# adj=pd.read_csv(r'C:\Users\admin\Desktop\My_master_piece_LOL\data\SZ\sz_adj.csv')
Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(data_csv.values)

#%%
"""divide data"""

trainX, trainY, testX, testY = divide_data(data=Norm_TS, train_size=train_size,
                                           seq_len=seq_len, pre_len=pre_len)
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()
# trainX, trainY = trainX.float(), trainY.float()
# testX, testY = testX.float(), testY.float()

adj = sp.coo_matrix(adj)
values = adj.data
indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
edge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式
edge_index=torch.LongTensor(edge_index).to(device)


edge_weight=np.ones(edge_index.shape[1])
edge_weight=torch.FloatTensor(edge_weight).to(device)

train_dataset = Data.TensorDataset(trainX, trainY)
test_dataset = Data.TensorDataset(testX, testY)

train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=False)


#%%
# net = RecurrentGCN(node_features=24, filters=32).to(device)
# net = GCN_LSTM(node_features=24, input_size=32, output_size=1).to(device)
# net = my_TGCN2(in_channels=24, out_channels=32,batch_size=batch_size).to(device)
net = ASTCG_net(in_channels=seq_len, embed_channels=32,
                out_channels=pre_len,adj_matrix=adj_matrix.values).to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.98)
loss_func = nn.MSELoss()



best_val_loss = float("inf")
best_model = None
#           train
train_loss_all = []
net.train()  # Turn on the train mode
total_loss = 0.

for epoch in range(500):

    cost = 0
    train_loss = 0
    train_num = 0
    for step, (x, y) in enumerate(train_loader):

        time_start = time.time()

        optimizer.zero_grad()


        x, y = x.to(device), y.to(device)  #batch train

        pre_y = net(x,edge_index.to(device),edge_weight.to(device))

        loss = loss_func(pre_y, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_num += x.size(0)

        time_end = time.time()
        time_c = (time_end - time_start)*10

        total_loss += loss.item()
        log_interval = int(len(trainX) / batch_size / 5)
        if (step + 1) % log_interval == 0 and (step + 1) > 0:
            cur_loss = total_loss / log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | '
                  'loss {:5.5f} | time {:8.2f}'.format(
                epoch, (step + 1), len(trainX) // batch_size, scheduler.get_last_lr()[0],
                cur_loss, time_c))
            total_loss = 0

    if (epoch + 1) % 5 == 0:
        print('-' * 89)
        print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
        print('-' * 89)
        train_loss_all.append(train_loss / train_num)

    if train_loss < best_val_loss:
        best_val_loss = train_loss
        best_model = net

    scheduler.step()

#%%
# best_model.eval()  # 转换成测试模式
# pre_data=[]
# for step, (x, y) in enumerate(test_loader):
#     pred_s=best_model(x.squeeze().to(device),edge_index,edge_weight)
#     pre_data.append(pred_s.data.cpu().numpy())
# all_simu=Normalization.inverse_transform(np.array(pre_data).squeeze())
# all_real=Normalization.inverse_transform(testY.data.numpy().squeeze())
#%% FOR BATCH
best_model.eval()  # 转换成测试模式
pred = best_model(testX.float().to(device),edge_index,edge_weight)
Norm_pred = pred.data.cpu().numpy()
all_simu=Normalization.inverse_transform(Norm_pred[:,:,-1])
all_real=Normalization.inverse_transform(testY.data.numpy()[:,:,-1])
#%%  ALL STEPS

Metric=[]
for i in range(all_simu.shape[0]):
    MAE, RMSE, MAPE, R2 = evaluation(all_real[i, :], all_simu[i, :])
    Metric.append([MAE, RMSE, MAPE,R2])

M = np.mean(np.array(Metric), axis=0)
M_sec = pd.DataFrame(Metric)

print(M)


#%%
# best_model.eval()  # 转换成测试模式
# pred = best_model(testX.float().to(device),edge_index,edge_weight)
# Norm_pred = pred.data.cpu().numpy()
# all_simu=Normalization.inverse_transform(Norm_pred.squeeze())
# all_real=Normalization.inverse_transform(testY.data.numpy().squeeze())
# Metric=[]
# for i in range(all_simu.shape[1]):
#     MAE, RMSE, MAPE, R2 = evaluation(all_real[:, i], all_simu[:, i])
#     Metric.append([MAE, RMSE, MAPE, R2])
#
# M = np.mean(np.array(Metric), axis=0)
# M_sec = pd.DataFrame(Metric)
# print(M)
# #%%
# import matplotlib.pyplot as plt
# plt.plot(all_simu[:,1],'r')
# plt.plot(all_real[:,1],'b')
# plt.show()
# #%%
# ERR_ours1=(all_real-all_simu).reshape(-1,1)
# ERR_ours2=(all_real-all_simu).reshape(-1,1)
# ERR_ours4=(all_real-all_simu).reshape(-1,1)
# #%%
# np.savez('3-Unfixed sampling/ERR_ours',
#          array1=ERR_ours1, array2=ERR_ours2, array3=ERR_ours4)