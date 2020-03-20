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

def upsample(img, k=2, out=None):
    nc, (h, w), e = img.shape[:-2], img.shape[-2:], 1e-4
    if out is None:
        out = np.zeros(nc+(h*k, w*k), dtype=img.dtype)
    
    rs = np.linspace(e,h-1-e, h*k, dtype=np.float32)
    cs = np.linspace(e,w-1-e, w*k, dtype=np.float32)
    ra = np.floor(rs).astype(np.uint32)
    ca = np.floor(cs).astype(np.uint32)
    rs -= ra
    cs -= ca

    outcol = out.reshape((-1, h*k, w*k))
    imgcol = img.reshape((-1, h, w))
    for i, o in zip(imgcol, outcol):
        _resize(i, k, ra, rs, 1-rs, ca, cs, 1-cs, o)
    return out

if __name__ == '__main__':
    from time import time
    from skimage.data import astronaut
    img = astronaut().transpose(2,0,1).astype(np.float32)

    # firs time to jit
    resize(img, 2)
    start = time()
    rst = resize(img, 2)
    print('\nmine v1', time()-start)
