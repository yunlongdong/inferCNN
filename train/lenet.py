import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import savemat
from time import time


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1   = nn.Linear(784, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.log_softmax(out, -1)
        return out

    def export_mat(self, fn='lenet.mat'):
        print(self.parameters)
        mat = {}
        for idx, param in enumerate(self.parameters()):
            print(param.data.shape)
            mat[str(idx)] = param.data.detach().numpy()

        savemat(fn, mat)




if __name__ == "__main__":

    net = LeNet()
    x = torch.FloatTensor(1, 1, 28, 28).zero_() + 1 
    net(x)
    net.export_mat()
    start = time()
    print(net(x))
    print('torch time:', time()-start) 
