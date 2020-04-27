import numpy as np


def CS_OMP(y, A, K):
    # y = A * theta
    y_rows, y_cols = y.shape
    if y_rows < y_cols:
        y = y.T
    M, N = A.shape
    theta = np.zeros([N])
    At = np.zeros([M, K])
    pos_theta = np.zeros(K)
    r_n = y
    for ii in range(K):
        product = np.dot(A.T, r_n)
        pos = np.argmax(abs(product))
        # print(pos)
        # print(pos)
        At[:, ii] = A[:, pos]
        pos_theta[ii] = pos
        # print(A[:, pos])
        A[:, pos] = np.zeros(M)
        # theta_ls = (At(:, 1:ii)'*At(:,1:ii))^(-1)*At(:,1:ii)' * y;
        Att = At[:, range(ii+1)]
        # print(Att)
        theta_ls = np.dot(np.dot(np.linalg.inv(np.dot(Att.T, Att)), Att.T), y)
        r_n = y - np.dot(Att, theta_ls)
    print(np.linalg.norm(r_n))