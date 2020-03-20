import numpy as np
from numba import njit
from math import floor

@njit
def _resize(img, k, ra, rs, _rs, ca, cs, _cs, out):
    h, w = img.shape
    for r in range(h*k):
        rar = ra[r]
        rbr = rar + 1
        rsr = rs[r]
        _rsr = _rs[r]
        for c in range(w*k):
            cac = ca[c]
            cbc = cac + 1
            rra = img[rar,cac]*_rsr
            rra += img[rbr,cac]*rsr
            rrb = img[rar,cbc]*_rsr
            rrb += img[rbr,cbc]*rsr
            rcab = rra * _cs[c] + rrb * cs[c]
            out[r,c] = rcab

def upsample(img, k, out=None):
    nc, (h, w) = img.shape[:-2], img.shape[-2:]
    if out is None:
        out = np.zeros(nc+(h*k, w*k), dtype=img.dtype)
    rs = np.linspace(-0.5+0.5/k,h-0.5-0.5/k, h*k, dtype=np.float32)
    cs = np.linspace(-0.5+0.5/k,w-0.5-0.5/k, w*k, dtype=np.float32)
    np.clip(rs, 0, h-1, out=rs)
    np.clip(cs, 0, w-1, out=cs)
    ra = np.floor(rs).astype(np.uint32)
    ca = np.floor(cs).astype(np.uint32)
    np.clip(ra, 0, h-1.5, out=ra)
    np.clip(ca, 0, w-1.5, out=ca)
    rs -= ra; cs -= ca; 
    outcol = out.reshape((-1, h*k, w*k))
    imgcol = img.reshape((-1, h, w))
    for i, o in zip(imgcol, outcol):
        _resize(i, k, ra, rs, 1-rs, ca, cs, 1-cs, o)
    return out

if __name__ == '__main__':
    from time import time
    from skimage.data import astronaut, camera
    img = astronaut().transpose(2,0,1).astype(np.float32)
    img = camera().astype(np.float32)
    img = np.array([[1,2],[3,4]], dtype=np.float32)
    # firs time to jit
    resize(img, 4)
    start = time()
    for i in range(10):
        rst1 = resize(img, 2)
    print('mine v1', time()-start)

    import cv2
    start = time()
    for i in range(10):
        rst2 = cv2.resize(img, (4, 4))
    print('cv', time()-start)
    
