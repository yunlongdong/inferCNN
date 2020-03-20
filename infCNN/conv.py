import numpy as np
from time import time
try: from numba import njit
except: njit = None

if njit is None: print('install numba can double speed!')

def neighbors(shape, core):
    shp = [slice(0,i) for i in core]
    idx = np.mgrid[tuple(shp)]
    idx = idx.reshape((len(core),-1))
    offset = np.array(core)//2
    offset[0] = 0
    idx -= offset.reshape((-1,1))
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx.T, acc[::-1])

def jit_fill_col(pdimg, idx, colimg):
    s = 0
    for i in range(len(pdimg)):
        if pdimg[i]&1==0: continue
        for j in idx:
            colimg[s] = pdimg[i+j]
            s += 1
    return colimg

def fill_col(pdimg, idx, colimg):
    rc = np.where(pdimg&1)[0]
    rc = rc.reshape((-1,1))+idx
    colimg[:] = pdimg[rc.ravel()]
    return colimg

if not njit is None: fill_col = njit(jit_fill_col)

def conv(img, core, stride=(1,1), buf=[np.zeros(1, dtype=np.int32)]):
    # new the col_img, if needed
    strh, strw = stride
    cimg_w = np.cumprod(core.shape[1:])[-1]
    n,c,h,w = img.shape
    cimg_h = n*(h//strh)*(w//strw)
    if len(buf[0])<cimg_h*cimg_w:
        buf[0] = col_img = np.zeros(cimg_h*cimg_w, dtype=np.int32)
    else:
        col_img = buf[0][:cimg_h*cimg_w]
        col_img[:] = 0
    # mark where need
    iimg = img.view(dtype=np.int32)
    iimg &= 0xfffffffe
    iimg[:,0,::strh,::strw] |= 1
    
    # ravel the image
    n,c,h,w = np.array(core.shape)
    shp = ((0,0),(0,0),(h//2,h//2),(w//2,w//2))
    pdimg = np.pad(iimg, shp, 'constant', constant_values=0)
    nbs = neighbors(pdimg.shape[1:], core.shape[1:])
    fill_col(pdimg.ravel(), nbs, col_img)
    col_img = col_img.view(np.float32)
    col_img = col_img.reshape((cimg_h, cimg_w))
    
    # dot
    col_core = core.reshape((core.shape[0],-1))
    rst = col_core.dot(col_img.T)
    ni, ci, hi, wi = img.shape
    return rst.reshape((ni, n, hi//strh, wi//strw))

if __name__ == '__main__':
    from skimage.data import camera
    import matplotlib.pyplot as plt
    from scipy.ndimage import convolve
    img = np.zeros((1, 3, 512, 512), dtype=np.float32)
    #img.ravel()[:] = np.arange(3*512*512)
    core = np.zeros((32, 3, 3, 3), dtype=np.float32)
    #core.ravel()[:] = np.arange(3*3*3*32)

    rst1 = conv(img, core, (1,1))
    start = time()
    rst1 = conv(img, core, (1,1))
    print('jit cost:', time()-start)

    start = time()
    rst2 = conv(img, core, (1,1))
    print('numpy cost:', time()-start)
