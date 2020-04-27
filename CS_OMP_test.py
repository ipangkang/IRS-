from CS_OMP import CS_OMP
import numpy as np
import random
K = 10
A = np.random.random([256, 1000])
theta = np.zeros([1000, 1])
# a = range(1000)
b = random.sample(range(1000), K)
for i in b:
    theta[i][0] = np.random.random()

y = np.dot(A, theta)
CS_OMP(y, A, K)