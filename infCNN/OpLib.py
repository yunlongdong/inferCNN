from .Core import Op 
from .im2col import im2col_indices
import numpy as np

    
class ReLU(Op):

    def __init__(self):
        super().__init__('ReLU')

    def forward(self, x):
        self.x = x
        return (x>0) * x


class Flatten(Op):

    def __init__(self):
        super().__init__('Flatten')

    def forward(self, x):
        n = x.shape[0]
        self.x = x
        return x.reshape((n, -1))


class Sigmoid(Op):

    def __init__(self):
        super().__init__('Sigmoid')

    def forward(self, x):
        self.x = x
        return 1/(1 + np.exp(-x))

class Softmax(Op):
    def __init__(self):
        super().__init__('Sofmax')

    def forward(self, X, axis=-1):
        eX = np.exp((X.T - np.max(X, axis=axis)).T)
        return (eX.T / eX.sum(axis=axis)).T




class MaxPool(Op):

    def __init__(self, size=2, stride=2):
        super().__init__('MaxPool_{}x{}s{}'.format(size, size, stride))
        self.size = size
        self.stride = stride

    def _pool_forward(self, X, pool_fun, size=2, stride=2):
        n, d, h, w = X.shape
        h_out = (h - size) / stride + 1
        w_out = (w - size) / stride + 1
    
        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')
    
        h_out, w_out = int(h_out), int(w_out)
    
        X_reshaped = X.reshape(n * d, 1, h, w)
        X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)
    
        out, pool_cache = pool_fun(X_col)
    
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3, 0, 1)
    
        cache = (X, size, stride, X_col, pool_cache)
    
        return out# , cache
    
    def maxpool(self, X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    def forward(self, x):
        return self._pool_forward(x, self.maxpool, self.size, self.stride)


relu = ReLU()
flatten = Flatten()
sigmoid = Sigmoid()
softmax = Softmax()
maxpool = MaxPool()
