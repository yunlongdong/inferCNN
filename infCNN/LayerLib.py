from .Core import Layer
from .conv import conv
import numpy as np


class Dense(Layer):

    def __init__(self, n_in, n_out):
        super().__init__('dense_{}_{}'.format(n_in, n_out))
        self.n_in, self.n_out = n_in, n_out
        self.K = np.zeros((n_out, n_in), dtype=np.float32)
        self.bias = np.zeros(n_out, dtype=np.float32)

    def forward(self, x):
        self.x = x
        y = x.dot(self.K.T)
        y += self.bias.reshape((1, -1))
        return y

    def load_from_torch(self, weights, bias):
        self.K = weights.copy()
        self.bias = bias.copy()


class Conv2d(Layer):

    def __init__(self, C_in, C_out, K_s, Stride):
        """
        Params:
            in channels, out channles, kernel size, stride
        """
        super().__init__("conv_{}_{}x{}x{}".format(C_in, C_out, K_s, K_s))
        self.c_in = C_in
        self.c_out = C_out
        self.k_s = K_s
        self.stride = Stride
        self.pad_size = int((K_s - 1)/2)
        self.K = np.zeros((C_out, C_in, K_s, K_s), dtype='float32')
        self.bias = np.zeros(C_out, dtype='float32')

    def forward(self, X):
        out = conv(X, self.K, (self.stride, self.stride))
        out += self.bias.reshape((1, -1, 1, 1))
        return out

    def load_from_torch(self, weights, bias):
        self.K = weights.copy()
        self.bias = bias.copy()


if __name__ == "__main__":
    # data = np.arange(16).reshape((1, 1, 4, 4))
    data = np.random.randn(2, 3, 200, 200)

    import torch.nn as nn
    import torch
    from time import time

    conv_torch = nn.Conv2d(3, 5, 3, 1, 1)
    conv = Conv2d(3, 5, 3, 1)
    # conv_torch.weight = nn.Parameter(torch.FloatTensor(np.arange(9).reshape(1, 1, 3, 3)))
    # conv_torch.bias = nn.Parameter(torch.FloatTensor(np.zeros(1)))
    w = conv_torch.weight.detach().numpy()
    b = conv_torch.bias.detach().numpy()

    start = time()
    out_torch = conv_torch(torch.FloatTensor(data)).detach().numpy()
    print('torch time:', time() - start)

    conv.load_from_torch(w, b)
    start = time()
    out = conv(data)
    print('acw time:', time() - start)

    print('torch:', out_torch.shape)
    print('acw:', out.shape)
    print(np.sum(out-out_torch))

    data = np.random.randn(2, 10)

    dense_torch = nn.Linear(10, 20)
    w = dense_torch.weight.detach().numpy()
    b = dense_torch.bias.detach().numpy()

    dense = Dense(10, 20)

    start = time()
    out_torch = dense_torch(torch.FloatTensor(data)).detach().numpy()
    print('torch time:', time() - start)

    dense.load_from_torch(w, b)
    start = time()
    out = dense(data)
    print('acw time:', time() - start)
    print(np.sum(out-out_torch))
