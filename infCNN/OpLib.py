from .Core import Op
import numpy as np
from .maxpool import maxpool
from .upsample import upsample


class ReLU(Op):

    def __init__(self):
        super().__init__('ReLU')

    def forward(self, x):
        self.x = x
        return (x > 0) * x


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


relu = ReLU()
flatten = Flatten()
sigmoid = Sigmoid()
softmax = Softmax()
maxpool = maxpool
upsample = upsample
