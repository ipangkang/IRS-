import random
import numpy as np
from pia_asp import PIA_ASP


# following the rule: K << M << N
K = 50
KK = 20
M = 256   # 256
N = 1000   # 1000
T = 1  # 100
# y = A * x
# x[t] & x[t+1] = KK
y = np.zeros([T, M, 1])


def main():
    A = gaussMatrix(T, M, N)
    x = setRelativity(N, T, K, KK)
    # print(A.shape)
    # print(x.shape)
    for t in range(T):
        y[t] = np.dot(A[t], x[t])
    # print(y.shape)
    PIA_ASP(y, A, sp=50)
# similarity
# output: x


def setRelativity(N, T, K, KK):
    array = range(N)
    x = np.zeros([T, N, 1])
    xx = x[0]
    pos = random.sample(array, K)
    for i in pos:
        xx[i] = random.random()
    # print(xx)

    for t in range(1, T):
        pos_old = random.sample(pos, KK)
        pos_new = set(array) - set(pos)
        pos_new = random.sample(pos_new, K-KK)
        pos = pos_new + pos_old
        # print(sorted(list(pos)))
        for i in pos:
            x[t][i][0] = random.random()
        # print(x[t])
    # print(x)
    return x


def gaussMatrix(T, M, N):
    mu, sigma = 1, np.sqrt(1/M)
    A = np.random.normal(mu, sigma, size=[T, M, N])
    # A = np.random.rand(T, M, N)
    # print(A)
    return A


def bernouliMatrix(T, M, N):
    p = [0.5, 0.5]
    val = [1/np.sqrt(M), -1/np.sqrt(M)]
    A = np.zeros([T, M, N])
    for t in range(T):
        for m in range(M):
            for n in range(N):
                A[t][m][n] = np.random.choice(val, p=p)
    return A


def toeplitxMatrix(M, N):
    pass


if __name__ == "__main__":
    main()