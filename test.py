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


'''
    N = S + E + I + R  # 总人数
    beta1 = 0.09  #感染者 单位感染人数：感染者的传染率*感染者者每天接触的人数
    beta2 = 0.6  #潜伏者 单位感染人数：潜伏着的传染率*潜伏者每天接触的人数
    theta1 =  I/ N * beta1
    theta2 = E / N * beta2
'''

# 递推最小二乘法(RLS)更新参数
def update_parameter(self, X, Y):
    theta_new = self + dot(dot(inv(dot(X.T, X)),
                               X[X.shape[0] - 1, :].T),
                           (Y[Y.shape[0] - 1] - dot(X[X.shape[0] - 1], self)))
    return theta_new


real_data = pd.read_csv('chinaDayList_2020_06_11.csv')
exisiting_suspect = real_data['total_suspect']
exisiting_confirm = real_data['existing confirmed']
exisiting_heal = real_data['total_heal']
datas = np.mat([exisiting_suspect, exisiting_confirm, exisiting_heal]).T
predict = 0
days = len(exisiting_confirm) - predict

N = 175000
model_datas = np.zeros((days + predict, 4))
model_datas[0] = [N, 54, 250, 25]
model_datas[1] = [N, 54, 250, 25]
R_parameter = np.mat([0])
I_parameter = np.mat([0, 0]).T
gammas = []
alphas = []
omegas = []
for i in range(2, days):
    R_parameter = update_parameter(R_parameter, np.mat(datas[:i, 1]), np.mat([datas[i, 2] - datas[i - 1, 2]]))

    gamma = R_parameter[0, 0]
    I_parameter = update_parameter(I_parameter, np.mat(np.c_[datas[:i, 0], datas[:i, 1] * -1]),
                                   (np.mat((datas[i, 1] - datas[i - 1, 1] * (1 - gamma))).T))

    alpha = I_parameter[0, 0]
    omega = I_parameter[1, 0]
    N = model_datas[i - 1, 0] + model_datas[i - 1, 1] + model_datas[i - 1, 2] + model_datas[i - 1, 3]  # 总人数
    beta1 = 0.09  # 感染者 单位感染人数：感染者的传染率*感染者者每天接触的人数
    beta2 = 0.6  # 潜伏者 单位感染人数：潜伏着的传染率*潜伏者每天接触的人数
    theta1 = model_datas[i - 1, 2] / N * beta1
    theta2 = model_datas[i - 1, 1] / N * beta2
    model_datas[i] = SEIR(model_datas[i - 1, 0], model_datas[i - 1, 1], model_datas[i - 1, 2], model_datas[i - 1, 3],
                          gamma, omega, alpha, theta1, theta2)
    gammas.append(gamma)
    alphas.append(alpha)
    omegas.append(omega)

csv = pd.DataFrame((np.mat((gammas, alphas, omegas)).T))
csv.to_csv('paramter.csv')


