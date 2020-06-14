import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv


# SEIR模型,S:易感者;E:潜伏者;I:感染者;R:康复者
def SEIR(S, E, I, R):
    T1 = 3  # 感染者者每天接触的人数
    T2 = 20  # 潜伏者每天接触的人数
    beta1 = 0.03  # 感染者的传染率
    beta2 = 0.03  # 潜伏着的传染率
    alpha = 0.4  # 潜伏者转换为感染者的概率
    omega = 0.03  # 感染死亡率
    gamma = 0.03  # 感染后康复率
    N = S + E + I + R  # 总人数
    S_new = S - I * T1 * S / N * beta1 - E * T2 * S / N * beta2
    E_new = E + I * T1 * S / N * beta1 + E * T2 * S / N * beta2 - E * alpha
    I_new = I + E * alpha - I * gamma - I * omega
    R_new = R + I * gamma
    return [S_new, E_new, I_new, R_new]


def update_theta(self, X, Y):
    theta_new = self + dot(dot(inv(dot(X.T, X)), X[X.shape[0], :]), (Y[Y.shape[0, :] - dot(X.shape[0].T, self)]))
    return theta_new


real_data = pd.read_csv('WuHan_time_series.csv')
day_confirm = real_data['city_dayConfirmedCount']
days = len(day_confirm)
N = 125500
datas = np.zeros((days, 4))
datas[0] = [N, 0, 3500, 0]
for i in range(1, days):
    datas[i] = SEIR(datas[i - 1, 0], datas[i - 1, 1], datas[i - 1, 2], datas[i - 1, 3])

x = range(days)
plt.plot(x, datas[:, 0], x, datas[:, 1], x, datas[:, 2], x, datas[:, 3], x, day_confirm)
plt.legend(('S', 'E', 'I', 'R', 'RealData'))
plt.show()
