from os.path import join as pjoin
import numpy as np
import os.path as osp
from imagepy.core.engine import Simple
from imagepy.core.mark import GeometryMark
from .net import LeNet

path = osp.abspath(osp.dirname(__file__))

class Plugin(Simple):
    title = 'mnist'
    note = ['8-bit', 'auto_snap', 'not_channel', 'preview']

    net = LeNet()
    net.load_mat(pjoin(path, 'lenet.mat'))

    def run(self, ips, imgs, para = None):
        marks = {'type':'layers', 'body':{}}
        for i, img in enumerate(imgs):
            tmp = img.copy().astype('float32')/255.0
            out = self.net(tmp[None, None, :, :])
            pred = np.argmax(out, -1)
            marks['body'][i] = {'type':'text', 'color':(255,0,0), 'body':(2, -2, str(pred))}
        ips.mark = GeometryMark(marks)
