import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
from torch import nn
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import  divide_data
import torch.utils.data as Data
import time

#%%
class GRU(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, pre_len):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=seq_len,  # 输入纬度   记得加逗号
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True,)
            # bidirectional=True)
        self.out = nn.Linear(hidden_size, pre_len)

    def forward(self, x):
        temp, _ = self.gru(x)
        s, b, h = temp.size()
        temp = temp.reshape(s * b, h)
        outs = self.out(temp)
        gru_out = outs.reshape(s, b, -1)
        return gru_out


class LSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, pre_len):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=seq_len,  # 输入纬度   记得加逗号
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5,
            batch_first=True)
        self.out = nn.Linear(hidden_size, pre_len)

    def forward(self, x):
        temp, _ = self.lstm(x)
        s, b, h = temp.size()
        temp = temp.reshape(s * b, h)
        outs = self.out(temp)
        lstm_out = outs.reshape(s, b, -1)
        return lstm_out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#%%  hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len=24
pre_len=4  #7:00:22:00
batch_size=32
train_size=3500 #80%
# set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#%%
data_csv = pd.read_csv(r'3-Unfixed sampling/periodic_sample.csv')
# data_csv = pd.read_csv(r'C:\Users\admin\Desktop\My_master_piece_LOL\data\SZ\sz_speed.csv')
Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(data_csv.values)
#%%
"""divide data"""

trainX, trainY, testX, testY = divide_data(data=Norm_TS, train_size=train_size,
                                           seq_len=seq_len, pre_len=pre_len)
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

train_dataset = Data.TensorDataset(trainX, trainY)
test_dataset = Data.TensorDataset(testX, testY)

train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size, shuffle=False)
#%%
'''init'''

GRU_net = GRU(seq_len=seq_len, hidden_size=32, num_layers=2, pre_len=pre_len).to(device)
LSTM_net = LSTM(seq_len=seq_len, hidden_size=32, num_layers=2, pre_len=pre_len).to(device)

net = LSTM_net

optimizer = torch.optim.AdamW(net.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.98)
loss_func = nn.MSELoss()
# num_paras=count_parameters(net)
#%%

best_val_loss = float("inf")
best_model = None
#           train
train_loss_all = []
net.train()  # Turn on the train mode
total_loss = 0.

for epoch in range(500):
    train_loss = 0
    train_num = 0
    for step, (x, y) in enumerate(train_loader):

        time_start = time.time()

        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        pre_y = net(x)

        loss = loss_func(pre_y, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)  # 梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_num += x.size(0)

        time_end = time.time()
        time_c = (time_end - time_start)*100

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

best_model.eval()  # 转换成测试模式
pred = best_model(testX.float().to(device))
Norm_pred = pred.data.cpu().numpy()
#%%  ALL STEPS NO R2
# all_simu=Normalization.inverse_transform(Norm_pred.squeeze())
# all_real=Normalization.inverse_transform(testY.data.numpy().squeeze())
all_simu=Normalization.inverse_transform(Norm_pred[:,:,-1])
all_real=Normalization.inverse_transform(testY.data.numpy()[:,:,-1])
Metric=[]
for i in range(all_simu.shape[0]):
    MAE, RMSE, MAPE, R2 = evaluation(all_real[i, :], all_simu[i, :])
    Metric.append([MAE, RMSE, MAPE, R2])

M = np.mean(np.array(Metric), axis=0)
M_sec = pd.DataFrame(Metric)

print(M)
# ALL SECTIONS

# Metric1=[]
# for i in range(all_simu.shape[1]):
#     MAE, RMSE, MAPE, R2 = evaluation(all_real[:, i], all_simu[:, i])
#     Metric1.append([MAE, RMSE, MAPE, R2])
#
# M1 = np.mean(np.array(Metric1), axis=0)
# M_sec1 = pd.DataFrame(Metric1)
# print(M1)

#%%
import matplotlib.pyplot as plt
plt.plot(all_simu[:,1],'r')
plt.plot(all_real[:,1],'b')
plt.show()
