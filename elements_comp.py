import numpy as np
import math as math
import pia_asp as pa
import dataset as dataset
from matplotlib import pyplot as plt
import copy
import time

# def G as channel gain with size (M * N)
# def Theta as IRS elements with size(N * N) and Theta = diag(theta_1, theta_2, ..., theta_N)
# def Hr as NLOS channel gain
# def Hd as LOS channel gain
# K as user number
# KK as active user number
# x as user array
# XL as Lagrange parameters with N length
# L as Lipschitz continuous gradient parameter
# rho as Lagrange multipliers
# here we first calculate d(yi) / d(theta_n)  i=1,...,M and n=1,...,N
# NN as user number
# K as active user number
M = 256
N = 40
NN = 1000
K = 50
L = 1e-10
rho = 1e-10
T = 1
KK = 20
Hr = np.random.uniform(low=1e-6, high=1e-4, size=[T, N, NN])
Hd = np.random.uniform(low=1e-13, high=1e-10, size=[T, M, NN])
G = np.random.uniform(low=1e-8, high=1e-6, size=[T, M, N])
XL = [8.5e-10] * N
theta_array = np.linspace(0, 2 * math.pi, 10)


def optimize(n, x, t, the, theta_w, theta_record, num):
    min_sum = 999
    min_theta = 0
    theta_matrix = np.zeros([num, num], dtype=complex)
    for i in range(num):
        theta_matrix[i][i] = math.cos(the[i]) + 1j * math.sin(the[i])

    for th in theta_array:
        theta_matrix[n][n] = th
        A = np.zeros([T, M, NN], dtype=complex)

        for t in range(T):
            A[t] = np.dot(np.dot(G[t, :, 0:num], theta_matrix), Hr[t, 0:num, :]) + Hd[t]  # print(A.shape)
        y = np.zeros([T, M, 1], dtype=complex)
        for t in range(T):
            y[t] = np.dot(A[t], x[t])
        # print(y.shape)
        x_array, resid_array = pa.PIA_ASP(y, A, sp=50)
        # print(resid_array)
        sum_else = XL[n] * abs(th - theta_w[n]) + (rho + L) / 2 * np.linalg.norm(th - theta_w[n])
        # sum_else = XL[n] * (th - theta_w[n]) + (rho + L) / 2 * np.linalg.norm(th - theta_w[n])
        # sum_else = 0
        # print(resid_array[-1])
        # print('sum_else', sum_else)
        if resid_array[-1] + sum_else < min_sum:
            # if sum_else < min_sum:
            min_sum = resid_array[-1] + sum_else
            # min_sum = sum_else
            min_theta = th
        # A = np.dot(np.dot(G, np.diag(theta)), Hr) + Hd
        # sumthetan = 0
        # for i in range(M):
        #     sumthetan += derivativeTheta(i, n, x, t) * derivativeTheta(i, n, x, t)
        #
        # sumthetan += XL[n] * (theta_wan - theta) + (rho + L) / 2 * np.linalg.norm(theta_wan - theta)
    return min_theta


def iteration(tt, num):
    theta = [0.1] * num
    theta_wan = [0.0] * num
    iteration_number = 10
    theta_record = [0.0] * num
    # # 1. get x
    x = dataset.setRelativity(NN, T, K, KK)

    # 2. get global theta
    for i in range(iteration_number):
        # print('迭代')
        for n in range(num):
            theta[n] = (XL[n] / rho + theta_wan[n]) % (2 * math.pi)

        # theta_wan = copy.copy(theta)
        # for n in range(N):
        #     theta[n] = XL[n] / rho + theta_wan[n]

        # 3. update global theta
        theta_tmp = [0] * num
        for n in range(len(theta)):
            theta_tmp[n] = optimize(n, x, tt, theta, theta_wan, theta_record, num) % (2 * math.pi)

        theta_wan = theta_tmp
        # print(theta_wan)
        # 4. Lagrange multipliers update
        for n in range(num):
            XL[n] += rho * (theta_wan[n] - theta[n])

        # print(theta_wan)
        # print('theta', theta)
        # theta_record = theta.copy()
        # print('XL', XL)
        # print()

    # np.save('theta.npy', theta)
    # np.save('x.npy', x)
    return theta, x
    # print(theta)
    # derivativeTheta()
    #   3.1 get derivative

    #   3.2 calculate


def comp(theta, xx, num):
    A_old = np.zeros([T, M, NN], dtype=complex)
    A_new = np.zeros([T, M, NN], dtype=complex)
    theta_matrix = np.diag(theta)
    for t in range(T):
        # A[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
        A_old[t] = Hd[t]
        # 这个应该是由theta_optimization 来决定的  A_new[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
        A_new[t] = np.dot(np.dot(G[t, :, 0:num], theta_matrix), Hr[t, 0:num, :]) + Hd[t]
        y_old = np.zeros([T, M, 1], dtype=complex)
        y_new = np.zeros([T, M, 1], dtype=complex)

        for t in range(T):
            y_old[t] = np.dot(A_old[t], xx[t])
            y_new[t] = np.dot(A_new[t], xx[t])
        before = time.time()
        x_array_old, resid_array_old = pa.PIA_ASP(y_old, A_old, sp=50)
        after = time.time()
        print('old', after-before)
        before = time.time()
        x_array_new, resid_array_new = pa.PIA_ASP(y_new, A_new, sp=50)
        after = time.time()
        print('new', after-before)
        resid_old = resid_array_old[-1]
        resid_new = resid_array_new[-1]

        # print('resid_old', resid_array_old[-1] / np.linalg.norm(A_old[t]))
        # print('resid_new', resid_array_new[-1] / np.linalg.norm(A_new[t]))
        # print('resid_old', resid_array_old[-1] / np.linalg.norm(y_old[t]))
        # print('resid_new', resid_array_new[-1] / np.linalg.norm(y_new[t]))
        return resid_array_old[-1] / np.linalg.norm(y_old[t]), resid_array_new[-1] / np.linalg.norm(y_new[t])


# 对比number of elements
if __name__ == "__main__":
    num_array = range(0,41,5)
    resid_new_record = []
    resid_old_record = []
    for num in num_array:
        theta, x = iteration(0, num=num)
        ol, ne = comp(theta, x, num=num)
        resid_new_record.append(ne)
        resid_old_record.append(ol)
    xaxis = np.arange(len(resid_new_record))
    plt.plot(num_array, resid_new_record)
    plt.plot(num_array, resid_old_record)
    print(resid_old_record)
    plt.show()
#

