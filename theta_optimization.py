import numpy as np
import math as math
import pia_asp as pa
import dataset as dataset
import copy

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


def derivativeTheta(i, n, x, t):
    sumHX = 0
    for j in range(K):
        sumHX += Hr[t][n][j] * x[t][j]
    return G[t][i][n] * sumHX


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


def comparison(x, theta_best):
    theta_matrix = [[0.0 + 1j * 0.0] * N for _ in range(N)]
    for i in range(N):
        theta_matrix[i][i] = math.cos(theta_best[i])  # + 1j * math.sin(the[i])

    A = np.zeros([T, M, NN])
    for t in range(T):
        A[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
    # print(A.shape)
    y = np.zeros([T, M, 1])
    for t in range(T):
        y[t] = np.dot(A[t], x[t])
    _, resid_array = pa.PIA_ASP(y, A, sp=50)
    r = resid_array[-1]

    print(r)

    theta_matrix = [[0.0] * N for _ in range(N)]
    for i in range(N):
        theta_matrix[i][i] = 1  # + 1j * math.sin(the[i])

    A = np.zeros([T, M, NN])
    for t in range(T):
        A[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
    # print(A.shape)
    y = np.zeros([T, M, 1])
    for t in range(T):
        y[t] = np.dot(A[t], x[t])
    _, resid_array = pa.PIA_ASP(y, A, sp=50)
    r_old = resid_array[-1]
    print('r_old', r_old)


def iteration(tt):
    theta = [0.1] * N
    theta_wan = [0.0] * N
    iteration_number = 100
    theta_record = [0.0] * N
    # # 1. get x
    x = dataset.setRelativity(NN, T, K, KK)
    # # print(x.shape)

    # # A = np.dot(np.dot(G, np.diag(theta)), Hr) + Hd
    # A = np.zeros([T, M, NN])
    # for t in range(T):
    #     A[t] = np.dot(np.dot(G[t], np.diag(theta)), Hr[t]) + Hd[t]
    # # print(A.shape)
    # y = np.zeros([T, M, 1])
    # for t in range(T):
    #     y[t] = np.dot(A[t], x[t])
    # # print(y.shape)
    # x_array = pa.PIA_ASP(y, A, sp=20)
    # for n in range(N):
    #     print('n',n)
    #     theta[n] = optimize(n,x,tt,theta)
    #     print(theta)

    # comparison(x, theta)
    # for i in range(len(x_array)):
    #     print(x_array[i])

    # 2. get global theta
    for i in range(iteration_number):
        # print('迭代')
        for n in range(N):
            theta[n] = (XL[n] / rho + theta_wan[n]) % (2 * math.pi)

        # theta_wan = copy.copy(theta)
        # for n in range(N):
        #     theta[n] = XL[n] / rho + theta_wan[n]

        # 3. update global theta
        theta_tmp = [0] * N
        for n in range(len(theta)):
            theta_tmp[n] = optimize(n, x, tt, theta, theta_wan, theta_record) % (2 * math.pi)

        theta_wan = theta_tmp
        # print(theta_wan)
        # 4. Lagrange multipliers update
        for n in range(len(XL)):
            XL[n] += rho * (theta_wan[n] - theta[n])
        print(theta_wan)
        print('theta', theta)
        theta_record = theta.copy()
        print('XL', XL)
        print()
    # print(theta)
    # derivativeTheta()
    #   3.1 get derivative

    #   3.2 calculate


if __name__ == "__main__":
    iteration(0)
