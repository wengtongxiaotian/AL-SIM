
import sys
sys.path.append('.')
import tool
import jax
import numpy
import jax.numpy as jnp
from utils.utils_metrics import ms_ssim
from utils.utils_phys import standard_psf_debye
from skimage.io import imread,imsave
# path = 
import pdb;pdb = pdb.set_trace
# img = imread(path)
# psf = jnp.array(img)[jnp.newaxis,jnp.newaxis]
def convolve(xin, k):
    x = xin.reshape([-1, 1, *xin.shape[2:]])
    y = jax.lax.conv_general_dilated(x, k, window_strides =(1, 1), padding='SAME')
    return y.reshape(xin.shape)
def rec_loss(x, rec):
    x_norm = (x-x.min()) / (x.max()-x.min())
    rec_norm = (rec-rec.min()) / (rec.max()-rec.min())
    return 0.875 * jnp.mean(jnp.abs(x - rec)) + 0.125 * (1 - ms_ssim(x_norm, rec_norm, win_size=5))
def simple_score(batch , net, params, rng, train):
    x,I,noise,psf,bg = batch
    psf_bg = standard_psf_debye(31,step = 10)
    
    D = convolve(I*x,psf)+noise + convolve(bg,psf_bg)
    # batch norm
    mean = D.mean(axis=(-2, -1), keepdims=True)
    var = D.var(axis=(-2, -1), keepdims=True)
    D = (D - mean) / (var + 1e-6) ** .5
    D = jax.lax.stop_gradient(D)
    # if not tool.global_dict.get('realdata',False):
    #     print('realdata')
    #     D = 
    res = net.apply({'params': params}, D , train, rngs={'dropout':rng})
    print(tool.printinfo())
    error = {}
    error["rec"] = rec_loss(x, res['rec'])
    error["nrmse"] = nrmse(x, res["rec"])
    error['rec_p'] = rec_loss(I,res['rec_p'])
    if tool.global_dict.get('onlypattern', False):
        error["loss"] = error['rec_p']
        print('mode: only pattern')
    elif tool.global_dict.get('onlyrecon',False):
        error["loss"] = error['rec']
    else:
        error["loss"] = error['rec']*0.50+error['rec_p']*0.50 #
    return error['loss'],(error,res)

def nrmse(x, y):
    return jnp.sqrt(jnp.mean((x - y)**2)) / jnp.sqrt(jnp.mean(x**2))


