import numpy as np
from scipy.io import loadmat
# infCNN layer
from infCNN import Conv2d, Dense
# infCNN op 
from infCNN import relu, flatten, maxpool, softmax 
from time import time

class LeNet():
    def __init__(self):
        self.conv1 = Conv2d(1, 6, 5, 1)
        self.conv2 = Conv2d(6, 16, 5, 1)
        self.fc1   = Dense(784, 120)
        self.fc2   = Dense(120, 84)
        self.fc3   = Dense(84, 10)

        self.weights = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        out = relu(self.conv1(x))
        # print(out.shape)
        out = maxpool(out)
        # print(out.shape)
        out = relu(self.conv2(out))
        # print(out.shape)
        out = maxpool(out)
        # print(out.shape)
        out = flatten(out)
        # print(out.shape)
        out = relu(self.fc1(out))
        # print(out.shape)
        out = relu(self.fc2(out))
        # print(out.shape)
        out = self.fc3(out)
        # print(out.shape)
        return softmax(out)

    def __call__(self, x):
        return self.forward(x)

    def load_mat(self, fn):
        data = loadmat(fn)
        w_len = len(data)-4
        print('load num of weights:', w_len)
        # minus 4 to skip the header
        for i in range(w_len//2+1):
            w = data[str(2*i)]
            b = data[str(2*i+1)]
            print(i, w.shape, b.shape)
            self.weights[i].load_from_torch(data[str(2*i)], data[str(2*i+1)][0, :])
        print('load done!')

if __name__ == "__main__":
    net = LeNet()
    data =  np.ones((1, 1, 28, 28), dtype='float32')
    x = np.ascontiguousarray(data, dtype=np.float32)
    print(data.shape, data.dtype)
    # net.load_mat('train/lenet.mat')
    print(net(x))
    start = time()
    print(net(x))
    print('infCNN LeNet time:', time()-start)

