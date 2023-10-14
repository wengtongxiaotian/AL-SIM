
import pdb
import jax
import numpy
import jax.numpy as jnp
from utils.utils_metrics import ms_ssim
from utils.utils_phys import standard_psf_debye, pattern_generation, get_Ks
def convolve(xin, k):
    x = xin.reshape([-1, 1, *xin.shape[2:]])
    y = jax.lax.conv_general_dilated(x, k, window_strides =(1, 1), padding='SAME')
    return y.reshape(xin.shape)
def rec_loss(x, rec):
    x_norm = (x-x.min()) / (x.max()-x.min())
    rec_norm = (rec-rec.min()) / (rec.max()-rec.min())
    return 0.875 * jnp.mean(jnp.abs(x - rec)) + 0.125 * (1 - ms_ssim(x_norm, rec_norm, win_size=5))
def simple_score(batch , net, params, rng, train):
    x,I,noise = batch
    psf_step = numpy.random.randint(20,40)
    psf_step = 40
    psf = standard_psf_debye(31,step = psf_step)
    D = jax.lax.stop_gradient(convolve(I*x,psf)+noise)
    # print(batch_stats)
    if train:
        res = net.apply({'params': params}, D , train, rngs={'dropout':rng})
    else:
        res = net.apply({'params': params}, D , train, rngs={'dropout':rng})
    error = {}
    error["rec"] = rec_loss(x, res['rec'])
    error["nrmse"] = nrmse(x, res["rec"])
    error['rec_p'] = rec_loss(I,res['rec_p'])
    error["loss"] = error['rec']*0.50+error['rec']*0.50 #_p
    return error['loss'],(error,res)

def nrmse(x, y):
    return jnp.sqrt(jnp.mean((x - y)**2)) / jnp.sqrt(jnp.mean(x**2))


