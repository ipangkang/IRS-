import numpy as np


def PIA_ASP(y, A, sp):
    set_prior = set()
    T, M, N = A.shape
    # M, N = A.shape
    KK = 5     # max iteration number
    R = {}
    set_pie = {}
    x_s = {}
    set_wan = {}
    set_out = {}
    x_out = {}
    set_s = {}
    x_wan = {}
    resid_array = []
    x_out_array = [[0] * N for _ in range(T)]
    for t in range(T):   # T
        addtwodim(x_wan, t, 0, {})
        i = 1
        addtwodim(set_wan, t, i-1, set())
        addtwodim(set_pie, t, i-1, set())
        addtwodim(R, t, i-1, y[t])
        s = sp + 1
        # print(s)
        if t == 1:
            sp = 0
        resid_record = 0
        while i <= KK:
            s_wan = sp - len(set_prior & set_pie[t][i-1])
            if s_wan >= 0:
                # print(A[t].T)
                # print(y[t])
                # print(np.dot(A[t].T, R[t][i-1]).reshape([N]))
                # print(R[t][i-1])
                # print(np.dot(A[t].T, R[t][i-1]).reshape([N])[list(set_prior)])
                pos = np.argsort(-abs(np.dot(A[t].T, R[t][i-1]).reshape([N])))[list(set_prior)]
                # matrix = np.dot(np.mat(A[t]).H, R[t][i-1])
                # pos = np.array(np.argsort(abs(np.dot(matrix, matrix.H)))[0])[0]
                # print(set_prior)
                # pos = np.argsort(-abs(np.dot(A[t].T, R[t][i-1]).reshape([N])[list(set_prior)]))
                # print(pos)
                gamma_a = set(pos[0:s_wan])
                # print('gamma', gamma_a)
                # gamma_a = set(np.argsort(-np.linalg.norm(np.dot(A[t].T, R[t][i-1])[:,list(set_prior)]))[0: s_wan])
            else:
                gamma_a = set()
            # print("gamma_a")
            # print(gamma_a)
            matrix = np.array(np.dot(np.mat(A[t]).H, R[t][i - 1])).reshape([N])
            # print(matrix)
            for jjj in range(len(matrix)):
                matrix[jjj] = matrix[jjj] * matrix[jjj].conjugate()
            # print(matrix)
            pos = np.argsort(-abs(matrix))
            # print(pos)
            # pos = np.argsort(-abs(np.dot(A[t].T, R[t][i - 1])).reshape([N]))
            gamma_b = set(pos[0:s])
            # print(gamma_b)
            # print("gamma_b")
            # print(gamma_b)
            gamma = set_pie[t][i-1] | gamma_a | gamma_b
            # print(gamma)
            # print(list(gamma))
            if len(gamma) <= M:
                # print(A.shape)
                At = A[t]
                At = At[:, list(gamma)]
                # print(At.shape)
                # print("At")
                # print(At)
            else:
                break
            # print(At.shape)
            # print(list(gamma))
            W_g = np.dot(np.dot(np.linalg.inv(np.dot(At.T, At)), At.T), y[t])
            # print(gamma)
            # print("np.linalg.norm(y[t]-np.dot(At, W_g))")
            # print(np.linalg.norm(y[t]-np.dot(At, W_g)))
            pos = np.argsort(-abs(W_g).reshape([len(gamma)]))[0:s]
            # print(pos)
            # print(s)
            # print(pos[0:s])
            # print("gamma")
            # print(gamma)
            # print(pos)
            # print(set(np.array(list(gamma))[pos[0:s]]))
            addtwodim(set_wan, t, i, set(np.array(list(gamma))[pos[0:s]]))     # support pruning
            # print(set_wan[t][i])

            # print(W_g)
            # print(gamma)
            # print(len(set_wan[t][i]))
            At = A[t]
            At = At[:, list(set_wan[t][i])]
            # print(At.shape)
            addtwodim(x_wan, t, i, np.dot(np.dot(np.linalg.inv(np.dot(At.T, At)), At.T), y[t]))
            resid = y[t] - np.dot(At, x_wan[t][i])
            addtwodim(R, t, i, resid)
            # print(np.linalg.norm(resid))
            if np.linalg.norm(R[t][i]) < np.linalg.norm(R[t][i-1]):
                addtwodim(set_pie, t, i, set_wan[t][i])
                x_s[t] = x_wan[t][i]
                set_s[t] = set_wan[t][i]
                i += 1
            else:
                x_s[t] = x_wan[t][i-1]
                set_s[t] = set_wan[t][i-1]
                s += 1
            # if np.linalg.norm(resid) < 1e-15:
            #     print(np.linalg.norm(resid))
            #     resid_array.append(np.linalg.norm(resid))
            #     set_out[t] = set_s[t]
            #     x_out[t] = x_s[t]
            #     set_prior = set_out[t]
            #     # print(set_out[t])
            #     for ii in range(len(set_out[t])):
            #         x_out_array[t][list(set_out[t])[ii]] = x_out[t][ii][0]
            #     # print(len(set_out[t]))
            #     break
            resid_record = np.linalg.norm(resid)
        resid_array.append(resid_record)
            # print(np.linalg.norm(resid))
        # print(len(set_s[t]))
        # print('=================')
    return x_out_array, resid_array


def addtwodim(dict, key_a, key_b, val):
    if key_a in dict:
        dict[key_a].update({key_b: val})
    else:
        dict.update({key_a: {key_b: val}})

