from .Core import Layer
from .im2col import im2col_indices
import numpy as np

class Dense(Layer):

    def __init__(self, n_in, n_out):
        super().__init__('dense_{}_{}'.format(n_in, n_out))
        self.n_in = n_in
        self.n_out = n_out
        self.K =  np.random.randn(n_out, n_in)
        self.bias = np.zeros(n_out)

    def forward(self, x):
        self.x = x
        w = self.K
        # the dense layer math: y = x*w
        y = w.dot(x.T)
        return y.T + self.bias.reshape((1, -1))

    def load_from_torch(self, weights, bias):
        self.K = weights
        self.bias = bias
    

class Conv2d(Layer):
    
    def __init__(self, C_in, C_out, K_s, Stride):
        """
        Params:
            in channels, out channles, kernel size, stride
        """
        super().__init__("conv_{}_{}x{}x{}".format(C_out, C_in, K_s, K_s))
        self.c_in = C_in
        self.c_out = C_out
        self.k_s = K_s
        self.stride = Stride
        self.pad_size = int((K_s -1)/2)
        self.K = np.zeros((C_out, C_in, K_s, K_s))
        self.bias = np.zeros(C_out)

    def forward(self, X):
        out = conv_forward(X, self.K, self.bias, stride=self.stride, padding=self.pad_size)
        return out

        
    def load_from_torch(self, weights, bias):
        self.K = weights
        self.bias = bias


def conv_forward(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col
    out = (out.T + b[:]).T
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out# , cache




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
