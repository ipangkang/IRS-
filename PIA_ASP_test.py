from pia_asp import PIA_ASP
import numpy as np
import random

T = 100
# y = np.random.random([T, 256, 1])
# A = np.random.random([T, 256, 1000])
sp = 10
K = 10
A = np.random.random([T, 256, 1000])
theta = np.zeros([T, 1000, 1])
# a = range(1000)
b = random.sample(range(1000), K)
for i in b:
    for t in range(T):
        theta[t][i][0] = np.random.random()
# A = np.array([[1, 2, 9, 10, 0, 1, 3, 6, 10, 9], [4, 3, 23, 2, 1, 4, 5, 6, 3, 3], [5, 6, 1, 5, 0, 1, 1, 0, 3, 1],
# [7, 3, 5, 9, 0, 1, 3, 1, 0, 0], [8, 8, 8, 8, 1, 3, 0, 3, 1, 0]])
y = np.zeros([T, 256, 1])
for t in range(T):
    y[t] = np.dot(A[t], theta[t])
print(y.shape)
PIA_ASP(y, A, sp)

# y = A \theta
# T is time slot
#
