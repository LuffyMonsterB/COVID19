import random
import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy import dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 最小二乘法求系数
def least_square(X, Y):
    X = np.mat(X)
    Y = np.mat(Y)
    beta = dot(dot(inv(dot(X.T, X)), X.T), Y)
    return beta


# 回归残差检验
def regression_residual_diagnosis(y_original, y_regression):
    y_original = np.mat(y_original)
    y_regression = np.mat(y_regression)
    SSE = np.sum(np.square(y_regression - y_original), axis=0)
    return SSE


# 导入excel表格数据
data = pd.read_excel('.//data//西瓜数据集.xlsx')
data = pd.DataFrame(data).to_numpy()

# 构造数据集X矩阵和Y矩阵。并在X中添加常数项
x = np.mat(data[:, 1:3])
constant = np.ones((x.shape[0], 1))
X = np.c_[constant, x]
Y = np.mat(data[:, 3]).T

# 从数据集中随机取出10个样本求解系数
# 记录SSE小于0.3的参数集
counts = 10
betas = np.zeros((100, 3))
count = 0
while 1:
    train_x = np.zeros((counts, 3))
    train_y = np.zeros((counts, 1))
    for i in range(counts):
        index = random.randint(0, X.shape[0] - 1)
        train_x[i] = X[index, 0:3]
        train_y[i] = Y[index, 0]
    # 用最小二乘法算出系数集β
    beta = least_square(train_x, train_y)
    # 计算出回归得到的拟合值
    y_regression = dot(X, beta)
    # 残差检验
    SSE = regression_residual_diagnosis(Y, y_regression)
    if SSE > 0.30:
        continue
    else:
        betas[count] = beta.T
        count += 1
        if count == 100:
            break
# 对残差检验结果良好的参数集取平均，并进行标准化后用于分析
beta_avg = np.mean(betas, axis=0)
SSE = regression_residual_diagnosis(Y, dot(X, beta_avg).T)
standard_beta = np.mean((betas.T - np.mean(betas.T, axis=0)) / np.std(betas.T, axis=0), axis=1)
# 结果打印
print('多元线性回归模型：y=' + str(beta_avg[0]) + '+' + str(beta_avg[1]) + 'x1+' + str(beta_avg[2]) + 'x2')
print('参数集β：' + 'β0=' + str(beta_avg[0]) + ' β1=' + str(beta_avg[1]) + ' β2=' + str(beta_avg[2]))
print('残差平方和SSE=' + str(SSE))
print('标准化参数集β为：' + str(standard_beta))

# 用三维散点图展示原始数据
x1 = data[:, 1].T
x2 = data[:, 2].T
y_original = data[:, 3].T
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1, x2, y_original, c='g', label='original')
ax.legend(loc='best')
# 用二元函数图表示二元线性回归模型
X_r, Y_r = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
Z_r = beta_avg[0] + beta_avg[1] * X_r + beta_avg[2] * Y_r
ax.plot_wireframe(X_r, Y_r, Z_r, rstride=1, cstride=1)
ax.view_init(elev=28, azim=0)  # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()
