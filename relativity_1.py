import numpy as np
import math as math
import pia_asp as pa
import dataset as dataset
from matplotlib import pyplot as plt
import copy
import time
import scipy.io


def optimize(n, x, t, the, theta_w, theta_record):
    min_sum = 999
    min_theta = 0
    theta_matrix = np.zeros([N, N], dtype=complex)
    for i in range(N):
        theta_matrix[i][i] = math.cos(the[i]) + 1j * math.sin(the[i])

    for th in theta_array:
        theta_matrix[n][n] = th
        A = np.zeros([T, M, NN], dtype=complex)
        for t in range(T):
            A[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
        # print(A.shape)
        y = np.zeros([T, M, 1], dtype=complex)
        for t in range(T):
            y[t] = np.dot(A[t], (x[t] + sigma[t]))
        np.savetxt('/Users/xfn/Downloads/cvx/projects/data/sigma.csv', sigma[0], delimiter=',')
        # print(y.shape)
        x_array, resid_array = pa.PIA_ASP(y, A, sp=0)
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


def iteration(tt):
    theta = [0.1] * N
    theta_wan = [0.0] * N
    iteration_number = 10
    theta_record = [0.0] * N
    # # 1. get x
    x = dataset.setRelativity(NN, T, K, KK)
    for i in range(iteration_number):
        for n in range(N):
            theta[n] = (XL[n] / rho + theta_wan[n]) % (2 * math.pi)

        theta_tmp = [0] * N
        for n in range(len(theta)):
            theta_tmp[n] = optimize(n, x, tt, theta, theta_wan, theta_record) % (2 * math.pi)

        theta_wan = theta_tmp
        # print(theta_wan)
        # 4. Lagrange multipliers update
        for n in range(len(XL)):
            XL[n] += rho * (theta_wan[n] - theta[n])

        # print(theta_wan)
        # print('theta', theta)
        theta_record = theta.copy()
        # print('XL', XL)
        # print()
    # np.save('theta.npy', theta)
    # np.save('x.npy', x)
    return theta, x

def comp(theta, xx):
    resid_old = []
    resid_new = []
    A_old = np.zeros([T, M, NN], dtype=complex)
    A_new = np.zeros([T, M, NN], dtype=complex)
    theta_matrix = np.diag(theta)
    for t in range(T):
        # A[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
        A_old[t] = Hd[t]
        A_new[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
        # print(A_new[t].shape)
        y_old = np.zeros([T, M, 1], dtype=complex)
        y_new = np.zeros([T, M, 1], dtype=complex)

        for t in range(T):
            y_old[t] = np.dot(A_old[t], xx[t])
            y_new[t] = np.dot(A_new[t], xx[t])

        x_array_old, resid_array_old = pa.PIA_ASP(y_old, A_old, sp=0)
        x_array_new, resid_array_new = pa.PIA_ASP(y_new, A_new, sp=0)

        print('resid_old', resid_array_old[-1] / np.linalg.norm(A_old[t]))
        print('resid_new', resid_array_new[-1] / np.linalg.norm(A_new[t]))

        return resid_array_old[-1] / np.linalg.norm(A_old[t]), resid_array_new[-1] / np.linalg.norm(A_new[t])


if __name__ == "__main__":
    # user_nums = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    user_num = 20
    resid_new = []
    resid_old = []

    # user_nums = [50, 100, 150, 200]
    M = 4
    N = 3
    # L = 1e-12
    # rho = 1e-12
    L = 1e-20
    rho = 1e-20
    T = 1
    # k_array = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    k_array = [2]
    sigma = 1e-17 * np.random.random([T, user_num, 1])

    theta = 0
    NN = user_num
    Hr = np.random.uniform(low=1e-6, high=1e-4, size=[T, N, NN])
    Hd = np.random.uniform(low=1e-10, high=1e-9, size=[T, M, NN])
    G = np.random.uniform(low=1e-6, high=1e-4, size=[T, M, N])
    x = 0
    for k in k_array:
        NN = user_num
        K = k
        KK = int(0.5 * K)
        # XL = [1e-12] * N
        XL = [1e-20] * N
        theta_array = np.linspace(0, 2 * math.pi, 10)

        theta, x = iteration(0)
        resid_o, resid_n = comp(theta, x)
        resid_old.append(0.1 * resid_o)
        resid_new.append(0.1 * resid_n)
        np.savetxt('/Users/xfn/Downloads/cvx/projects/data/x.csv', x[0], delimiter=',')
    print('theta', theta)
    theta_out = np.zeros([len(theta), len(theta)], dtype=complex)
    for i in range(len(theta)):
        theta_out[i][i] = math.cos(theta[i]) + 1j * math.sin(theta[i])
    AA = np.dot(np.dot(G[0], theta_out), Hr[0]) + Hd[0]

    print('resid_old',resid_old)
    print('resid_new',resid_new)

    scipy.io.savemat('/Users/xfn/Downloads/cvx/projects/data/matData.mat', {'theta_out': theta_out, 'AA': AA, 'x': x[0], 'sigma': sigma[0]})  # 写入mat文件
