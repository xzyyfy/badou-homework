import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def diy_lstm(x, input_size, hidden_size, baise=True):
    weight_h = np.zeros(shape=(1, hidden_size))
    weight_gate = np.array(np.random.random(size=(4, hidden_size + input_size)))
    output = []
    last_c_t = np.zeros(shape=hidden_size + input_size)
    for x_t in x:
        # 增加维度
        x_t = x_t[np.newaxis, :]
        hx = np.concatenate(weight_h, x_t)
        [f_t, i_t, pre_c_t, o_t] = sigmoid(np.dot(hx, weight_gate.T))
        g_c_t = np.tanh(pre_c_t)
        # 第一个时间步的c_t是怎么产生的呢???
        c_t = np.dot(f_t, c_t.T) + np.dot(i_t, g_c_t.T)
        h_t_next = o_t * np.tanh(c_t)
        output.append(h_t_next)
