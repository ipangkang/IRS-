import pia_asp as pa
import numpy as np
import math as math
import dataset as dataset

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
x = np.load('x.npy')
theta_opt = np.load('theta.npy')
theta_matrix = np.diag(theta_opt)

if __name__ == "__main__":
    # x = dataset.setRelativity(NN, T, K, KK)

    A_old = np.zeros([T, M, NN], dtype=complex)
    A_new = np.zeros([T, M, NN], dtype=complex)
    for t in range(T):
        # A[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
        A_old[t] = Hd[t]
        #这个应该是由theta_optimization 来决定的  A_new[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
        A_new[t] = np.dot(np.dot(G[t], theta_matrix), Hr[t]) + Hd[t]
        y_old = np.zeros([T, M, 1], dtype=complex)
        y_new = np.zeros([T, M, 1], dtype=complex)

        for t in range(T):
            y_old[t] = np.dot(A_old[t], x[t])
            y_new[t] = np.dot(A_new[t], x[t])

        x_array_old, resid_array_old = pa.PIA_ASP(y_old, A_old, sp=50)
        x_array_new, resid_array_new = pa.PIA_ASP(y_new, A_new, sp=50)

        print('resid_old', resid_array_old[-1])
        print('resid_new', resid_array_new[-1])
