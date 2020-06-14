import math
import matplotlib.pyplot as plt


def update_omega(time):
    a1 = 0.4902
    b1 = 10.01
    c1 = 2.977
    a2 = -2.236
    b2 = -15.46
    c2 = 12.77
    omega = a1 * math.exp(-((time - b1) / c1) ** 2) + a2 * math.exp(-((time - b2) / c2) ** 2)
    return omega


def update_alpha(time):
    a1 = 0.2172
    b1 = 9.961
    c1 = 2.535
    a2 = 4.865e+13
    b2 = -2867
    c2 = 500.8
    alpha = a1 * math.exp(-((time - b1) / c1) ** 2) + a2 * math.exp(-((time - b2) / c2) ** 2)
    return alpha


def update_parameter(time, a1, b1, c1, a2, b2, c2):
    return a1 * math.exp(-((time - b1) / c1) ** 2) + a2 * math.exp(-((time - b2) / c2) ** 2)


def update_gamma(time):
    a = 506.6
    b = 866.7
    c = -795.1
    return (a * math.atan(time + b) + c)


gammas = []
omegas = []
alphas = []

for i in range(100):
    # gamma = update_parameter(i, -0.006985, 52.04, 7.118, 0.05265, 63.41, 39.15)
    gamma = update_parameter(i,0.008389,2.631,2.54,0.00663,7.943,2.219)
    alpha = update_parameter(i, 0.2172, 9.961, 2.535, 4.865e+13, -2867, 500.8)
    omega = update_parameter(i, 0.4902, 10.01, 2.977, -2.236, -15.46, 12.77)

    gammas.append(gamma)
    alphas.append(alpha)
    omegas.append(omega)
x = range(100)
plt.plot(x, gammas, x, alphas, x, omegas)
plt.legend(('gamma', 'alpha', 'omega'))
# plt.plot(x, model_datas[:, 0], x, model_datas[:, 1], x, model_datas[:, 2], x, model_datas[:, 3], x, exisiting_confirm,x, exisiting_suspect, x, exisiting_heal)
# plt.legend(('S', 'E', 'I', 'R', 'confirm', 'suspect', 'heal'))
plt.show()
