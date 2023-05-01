# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import math
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
import xgboost as xgb
from sklearn import preprocessing
from matplotlib import pyplot as plt
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import divide_data
import warnings
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 12,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)
warnings.filterwarnings("ignore")

#%%
def Baselines(method,trainX,trainY,testX):
    # 预测
    if method == 'LR':  # 线性回归
        LR = LinearRegression()
        Norm_pred = LR.fit(trainX, trainY).predict(testX)
    elif method == 'MLP':  # Mlp回归
        MLP = MLPRegressor(hidden_layer_sizes=(16,32), activation='relu',
                           batch_size='auto',solver='adam', alpha=1e-04,
                           learning_rate_init=0.001, max_iter=300,beta_1=0.9,
                           beta_2=0.999, epsilon=1e-08)
        Norm_pred = MLP.fit(trainX, trainY).predict(testX)
    elif method == 'DT':  # 决策树回归
        DT = DecisionTreeRegressor()
        Norm_pred = DT.fit(trainX, trainY).predict(testX)
    elif method == 'SVR':  # 支持向量机回归
        SVR_MODEL = svm.SVR()
        Norm_pred = SVR_MODEL.fit(trainX, trainY).predict(testX)
    elif method == 'KNN':  # K近邻回归
        KNN = KNeighborsRegressor(10, weights="distance", leaf_size=30,
                                  algorithm='auto', p=1,)
                                  # metric='chebyshev')
        Norm_pred = KNN.fit(trainX, trainY).predict(testX)
    elif method == 'RF':  # 随机森林回归
        RF = RandomForestRegressor(n_estimators=100,max_depth=10,
                                   random_state=42,criterion='mse')
        Norm_pred = RF.fit(trainX, trainY).predict(testX)
    elif method == 'AdaBoost':  # Adaboost回归
        AdaBoost = AdaBoostRegressor(n_estimators=50)
        Norm_pred = AdaBoost.fit(trainX, trainY).predict(testX)
    elif method == 'XGB':  # XGB回归
        XGB_params = {'learning_rate': 0.1, 'n_estimators': 40,
                      'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                      'objective': 'reg:squarederror', 'subsample': 0.8,
                      'colsample_bytree': 0.8, 'gamma': 0,
                      'reg_alpha': 0.1, 'reg_lambda': 0.1}
        XGB = xgb.XGBRegressor(**XGB_params)
        Norm_pred = XGB.fit(trainX, trainY).predict(testX)
    elif method == 'GBRT':  # GBRT回归
        GBRT = GradientBoostingRegressor(n_estimators=100)
        Norm_pred = GBRT.fit(trainX, trainY).predict(testX)
    elif method == 'BR':  # Bagging回归
        BR = BaggingRegressor()
        Norm_pred = BR.fit(trainX, trainY).predict(testX)
    elif method == 'ETR':  # ExtraTree极端随机树回归
        ETR = ExtraTreeRegressor()
        Norm_pred = ETR.fit(trainX, trainY).predict(testX)

    return Norm_pred
# %% # 数据读取与预处理
data_csv = pd.read_csv(r'3-Unfixed sampling/periodic_sample.csv')
# data_csv = pd.read_csv(r'C:\Users\admin\Desktop\My_master_piece_LOL\data\SZ\sz_speed.csv')
# Normalization = preprocessing.MinMaxScaler()
# Norm_TS = Normalization.fit_transform(data_csv.values)
# %%set seed for reproductive 42 is the answer to the universe
seq_len=24
pre_len=4  #7:00:22:00
batch_size=32
train_size=3500 #80%
# set seed for reproductive 42 is the answer to the universe
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

#%%

trainX, trainY, testX, testY = divide_data(data=data_csv.values,
                                           train_size=train_size,
                                           seq_len=seq_len, pre_len=pre_len)

trainX, trainY = np.array(trainX).squeeze(), np.array(trainY).squeeze()
testX, testY = np.array(testX).squeeze(), np.array(testY).squeeze()
#%%
# Norm_pred = Baselines(method='XGB', trainX=trainX.reshape(trainX.shape[0],-1),
#                       trainY=trainY, testX=testX.reshape(testX.shape[0],-1))

#%%
all_simu=[]
all_real=[]
Metric=[]
for i in range(23):
    # Norm_pred = Baselines(method='MLP', trainX=trainX[:,:,i],
    #                       trainY=trainY[:,i], testX=testX[:,:,i])
    Norm_pred = Baselines(method='MLP', trainX=trainX[:,:,i],
                          trainY=trainY[:,-1,i].reshape(-1,1), testX=testX[:,:,i])
    all_simu.append(Norm_pred)
    all_real.append(testY[:,-1,i])
    # all_real.append(testY[:,i])
    # MAE, RMSE, MAPE, R2 = evaluation(testY[:,i], Norm_pred)
    MAE, RMSE, MAPE, R2 = evaluation(testY[:,-1, i], Norm_pred)
    Metric.append([MAE, RMSE, MAPE, R2])



M = np.mean(np.array(Metric), axis=0)
M_sec = pd.DataFrame(Metric)

print(M)
all_simu=np.vstack(all_simu)
all_real=np.vstack(all_real)
#%%

Metric=[]
for i in range(all_simu.shape[0]):
    MAE, RMSE, MAPE, R2 = evaluation(all_real[i, :], all_simu[i, :])
    Metric.append([MAE, RMSE, MAPE, R2])

M = np.mean(np.array(Metric), axis=0)
M_sec = pd.DataFrame(Metric)

print(M)