import numpy as np


# up 2x featuremap, NxCxHxW -> NxCx2Hx2W
def up2x(fm):
    shape2x = np.array(fm.shape)
    shape2x[-2:] *= 2

    fm2x = np.zeros(shape2x)
    fm2x[:, :, ::2, ::2] = fm
    fm2x[:, :, 1::2, 1::2] = fm
    
    return fm2x


# up 4x featuremap, NxCxHxW -> NxCx4Hx4W
def up4x(fm):
    fm4x = up2x(up2x(fm))
    return fm4x

if __name__ == "__main__":
    img = np.random.randn(2, 3, 100, 100)
    img2x = up4x(img)
    print(img.shape, img2x.shape)

    print(np.sum(img - img2x[:, :, ::4, ::4]))

    
