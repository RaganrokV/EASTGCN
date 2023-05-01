from math import sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def evaluation(real_p, sim_p):
    MAE = mean_absolute_error(real_p, sim_p)

    RMSE = sqrt(mean_squared_error(real_p, sim_p))

    if (real_p == 0).any():
        MAPE = np.median(np.abs((real_p - sim_p) / real_p))   # 数据有0用mdape
    else:
        MAPE = np.mean(np.abs((real_p - sim_p) / real_p))

    R2 = r2_score(real_p.squeeze(), sim_p.squeeze())
    return MAE, RMSE, MAPE, R2
