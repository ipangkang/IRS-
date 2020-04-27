import numpy as np
import random
import math
# implement weighted sum rate optimaization in the article:
# "Weighted Sum-Rate Optimization for Intelligent Reflecting Surface Enhanced Wireless Networks"
# Huayan Guo, Member, IEEE, Ying-Chang Liang, Fellow, IEEE, May 2019
N = 4
M = 4
K = 2
alpha = np.zeros([K, 1])
gamma = np.zeros([K, 1])

def initialize():
    set_w()
    set_theta()
    pass


def set_w():
    W = np.zeros([M, N], dtype=complex)
    for i in range(len(W)):
        for j in range(len(W[0])):
            W[i, j] = complex(random.random(), random.random())
    print(W)
    return W


# set theta = A e^(j*phi)
# amplitude A (0, 1]
# phase phi [0, 2 * pi)
def set_theta():
    theta_array = np.zeros([N, N], dtype=complex)
    for i in range(N):
        A = random.random()
        phi = random.uniform(0, 2 * math.pi)
        theta_array[i, i] = complex(A * math.cos(phi), A * math.sin(phi))
    print(theta_array)
    return theta_array


def optimize_part1():
    pass


def update_alpha():
    alpha = gamma
    pass


def update_beta():
    beta_k = 1  # equation 15
    pass


def update_w():
    w_k = 1 # equation 16
    pass


def update_e():
    pass


def update_lumbda():
    pass


def update_theta():
    pass


def update_gamma():
    for i in range(K):
        gamma[i] = 1    # equation 12

    print(gamma)
    pass


if __name__ == "__main__":
    set_theta()
    set_w()
