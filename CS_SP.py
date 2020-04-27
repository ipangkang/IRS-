import numpy as np


def CS_SP(y, A, K):
    # y = Phi * x
    # x = Psi * theta
    # y = Phi * Psi * theta
    # A = Phi * Psi, y = A * theta
    # % K is the sparsity level
    y_rows, y_cols = y.shape
    if y_rows < y_cols:
        y = y.T
    M, N = A.shape
    theta = np.zeros([N, 1])
    pos_theta = []
    r_n = y
    for k in range(K):
        print(A.T)
        # print(r_n)
        product = np.dot(A.T, r_n)
        # print(product)
        p = product.reshape([N])
        print("p")
        print(p)
        pos = np.argsort(-abs(p))
        # print(val)
        print("pos")
        print(pos)
        Js = pos[0: K]
        Is = []
        print("Js")
        print(Js)
        for i in Js:
            Is.append(i)
        for i in pos_theta:
            Is.append(i)
        Is = np.array(Is)
        Is.sort()
        print("Is")
        print(Is)
        if len(Is) <= M:
            At = A[:, Is]
            print("At")
            print(At.shape)
            print(At)
        else:
            break
        #y = At * theta theta LeastSquare
        print("At.T")
        print(At.T)
        theta_ls = np.dot(np.dot(np.linalg.inv(np.dot(At.T, At)), At.T), y)    # 2*1
        print("theta_ls")
        print(theta_ls)
        l = len(theta_ls)
        p = np.reshape(theta_ls, [l])
        print("p")
        print(p)
        pos = np.argsort(-abs(p))
        print("pos")
        print(pos)
        pos_theta = Is[pos[0:K]]
        print("pos_theta")
        print(pos_theta)
        theta_ls = theta_ls[pos[0:K]]
        print("theta_ls")
        print(theta_ls)
        # pos = np.argsort(-abs(theta_ls))
        # print("new_pos")
        # print(pos)
        print("np.dot(At[:, pos[0:K]], theta_ls)")
        print(np.dot(At[:, pos[0:K]], theta_ls))
        print("At[:, pos[0:k]]")
        print(At[:, pos[0:K]])
        print("y")
        print(y)
        r_n = y - np.dot(At[:, pos[0:K]], theta_ls)
        print("r_n")
        print(r_n)
        print("r_n norm")
        print(np.linalg.norm(r_n))
        if np.linalg.norm(r_n) < 1e-6:
            print("k")
            print(k)
            break
    theta[pos_theta] = theta_ls
    print(theta)
    residual = np.linalg.norm(y - np.dot(A, theta))
    print(residual)
        # theta_ls = (At.T * At)^(-1)*At' * y
    # print(y)
    # print(y_rows)
    # print(y_cols)
