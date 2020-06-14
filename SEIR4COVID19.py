import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv


# SEIR模型,S:易感者 ; E:潜伏者 ; I:感染者 ; R:康复者
# gamma:康复率 ; omega:感染死亡率 ; alpha:潜伏者转换为感染者的概率 ; theta1:感染者的传染率 ; theta2:潜伏着的传染率
def SEIR(S, E, I, R, gamma, omega, alpha, theta1, theta2):
    S_new = S - S * theta1 - S * theta2
    E_new = E + S * theta1 + S * theta2 - E * alpha
    I_new = I + E * alpha - I * gamma - I * omega
    R_new = R + I * gamma
    return [S_new, E_new, I_new, R_new]


# 递推最小二乘法(RLS)更新参数
def update_parameter(self, X, Y):
    theta_new = self + dot(dot(inv(dot(X.T, X)),
                               X[X.shape[0] - 1, :].T),
                           (Y[Y.shape[0] - 1] - dot(X[X.shape[0] - 1], self)))
    return theta_new


# 正态分布下更新参数
def update_parameter(time, a1, b1, c1, a2, b2, c2):
    return a1 * math.exp(-((time - b1) / c1) ** 2) + a2 * math.exp(-((time - b2) / c2) ** 2)


# 残差检验
def residual_diagnosis(y_original, y_model):
    SSE = np.sum(np.square(y_original - y_model), axis=0)
    means = np.mat(np.mean(y_model))
    mean_mat = np.repeat(means, y_model.shape[0], axis=1).T
    R_square = 1 - SSE / np.sum(np.square(y_original - mean_mat))
    return R_square


real_data = pd.read_csv('chinaDayList_2020_06_11.csv')
exisiting_suspect = real_data['total_suspect']
exisiting_confirm = real_data['existing confirmed']
exisiting_heal = real_data['total_heal']
datas = np.mat([exisiting_suspect, exisiting_confirm, exisiting_heal]).T
predicit = 0
days = len(exisiting_confirm) + predicit

N = 85000
model_datas = np.zeros((days, 4))
model_datas[0] = [N, 54, 250, 25]
model_datas[1] = [N, 54, 250, 25]
R_parameter = np.mat([0])
I_parameter = np.mat([0, 0]).T

for i in range(1, days):
    gamma = update_parameter(i, 0.008389, 2.631, 2.54, 0.00663, 7.943, 2.219)
    alpha = update_parameter(i, 0.2172, 9.961, 2.535, 4.865e+13, -2867, 500.8)
    omega = update_parameter(i, 0.4902, 10.01, 2.977, -2.236, -15.46, 12.77)
    N = model_datas[i - 1, 0] + model_datas[i - 1, 1] + model_datas[i - 1, 2] + model_datas[i - 1, 3]  # 总人数
    beta1 = 0.2  # 感染者 单位感染人数：感染者的传染率*感染者者每天接触的人数
    beta2 = 0.7  # 潜伏者 单位感染人数：潜伏着的传染率*潜伏者每天接触的人数
    theta1 = model_datas[i - 1, 2] / N * beta1
    theta2 = model_datas[i - 1, 1] / N * beta2
    model_datas[i] = SEIR(model_datas[i - 1, 0], model_datas[i - 1, 1], model_datas[i - 1, 2], model_datas[i - 1, 3],
                          gamma, omega, alpha, theta1, theta2)

SSE_E = residual_diagnosis(datas[:, 0], np.mat(model_datas[:, 1]).T)
SSE_I = residual_diagnosis(datas[:, 1], np.mat(model_datas[:, 2]).T)
SSE_R = residual_diagnosis(datas[:, 2], np.mat(model_datas[:, 3]).T)
print(SSE_E, SSE_I, SSE_R)
print(model_datas)
x = range(days)
x_real = range(days - predicit)
date = pd.date_range('2020-1-20', '2020-6-10')

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(x, model_datas[:, 0], x, model_datas[:, 1], x, model_datas[:, 2], x, model_datas[:, 3], x_real,
         exisiting_confirm,
         x_real, exisiting_suspect, x_real, exisiting_heal)
plt.legend(('S', 'E', 'I', 'R', 'confirm', 'suspect', 'heal'))
plt.xlabel('Date')
plt.ylabel('Number')
plt.show()
