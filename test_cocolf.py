import numpy as np
import skimage.io as io
import jax
xx = io.imread('/dataf/b/data/26366s_main_178.png')
yy = io.imread('/dataf/b/data/26366s_main_175.png')
def convolve(xin, k):
    x = xin.reshape([-1, 1, *xin.shape[2:]])
    y = jax.lax.conv_general_dilated(x, k, window_strides =(1, 1), padding='SAME')
    return y.reshape(xin.shape)
print(xx.shape)
xx = xx.reshape(3,256,3,256).transpose(0,2,1,3).reshape(1,9,256,256)
psf = io.imread('/dataf/b/_record/cocoall/np/tiffs/000073s_main_175.tif')
D1 = xx*yy[np.newaxis,np.newaxis]
D1 = D1/D1.max()
D = convolve(D1,psf[np.newaxis,np.newaxis])

io.imsave('tmp.tif',D)

