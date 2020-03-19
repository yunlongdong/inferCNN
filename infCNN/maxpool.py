import numpy as np
from time import time
from numba import njit

def neighbors(shape, core):
    shp = [slice(0,i) for i in core]
    idx = np.mgrid[tuple(shp)]
    idx = idx.reshape((len(core),-1))
    offset = np.array(core)//2
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx.T, acc[::-1])

@njit
def fill_col(pdimg, idx, colimg):
    s = 0
    for i in range(len(pdimg)):
        if pdimg[i]&1==0: continue
        for j in idx:
            colimg[s] = max(colimg[s], pdimg[i+j])
        s += 1
    return colimg

def maxpool(img, stride=(2,2)):
    strh, strw = stride
    n,c,h,w = img.shape
    cimg_h = n*c*(h//strh)*(w//strw)
    
    iimg = img.view(dtype=np.int32)
    iimg &= 0xfffffffe
    iimg[:,:,::strh,::strw] |= 1
    
    nbs = neighbors(img.shape[1:], (1,)+stride)
    shp = (n, c, h//strh, w//strw)
    colimg = np.zeros(shp, dtype=np.int32)
    fill_col(iimg.ravel(), nbs, colimg.ravel())
    return colimg.view(np.float32)

if __name__ == '__main__':
    img = np.zeros((1, 3, 512, 512), dtype=np.float32)

    
    start = time()
    rst1 = maxpool(img)
    print(time()-start)

    start = time()
    rst = maxpool(img)
    print('jit cost: x10', time()-start)
