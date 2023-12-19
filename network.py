from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Callable
class DND_SIM(nn.Module):
    """ 
        Deep Neural Decoding Structured Illumination Microscopy (DND-SIM)
    """
    features: int = 32
    @nn.compact
    def __call__(self, D, training=True):

        x = jnp.transpose(D, (0, 2, 3, 1))
        z1, z2, z3, z4_dropout, z5_dropout = Encoder(self.features)(x, training)
        y = Decoder(self.features, out=1)(z1, z2, z3, z4_dropout, z5_dropout)
        # y = x[...,:1]
        rec_x = jnp.transpose(y, (0, 3, 1, 2))
        # rec_p = D
        rec_p = Decoder(32, out=9)(z1, z2, z3, z4_dropout, z5_dropout)
        # rec_p = nn.activation.softmax(rec_p)*9 #when realdata and not only pattern it's will be nonsense
        rec_p = jnp.transpose(rec_p,(0,3,1,2))
        # result
        res = {
            "rec": rec_x,
            "rec_p":rec_p,
            "D":D,
        }
        return res


def convolve(xin, k):
    x = xin.reshape([-1, 1, *xin.shape[2:]])
    y = jax.lax.conv_general_dilated(x, k, window_strides =(1, 1), padding='SAME')
    return y.reshape(xin.shape)


class Encoder(nn.Module):
    features: int = 64
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, training):
        z1 = nn.Conv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        z1 = nn.relu(z1)
        z1 = nn.Conv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z1)
        z1 = nn.relu(z1)
        z1_pool = nn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

        z2 = nn.Conv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z1_pool)
        z2 = nn.relu(z2)
        z2 = nn.Conv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z2)
        z2 = nn.relu(z2)
        z2_pool = nn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

        z3 = nn.Conv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z2_pool)
        z3 = nn.relu(z3)
        z3 = nn.Conv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z3)
        z3 = nn.relu(z3)
        z3_pool = nn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

        z4 = nn.Conv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z3_pool)
        z4 = nn.relu(z4)
        z4 = nn.Conv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z4)
        z4 = nn.relu(z4)
        z4_dropout = nn.Dropout(0.5, deterministic=not training)(z4)
        z4_pool = nn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))

        z5 = nn.Conv(self.features * 16, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z4_pool)
        z5 = nn.relu(z5)
        z5 = nn.Conv(self.features * 16, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z5)
        z5 = nn.relu(z5)
        z5_dropout = nn.Dropout(0.5, deterministic=not training)(z5)
        # z5_dropout = nn.Conv(1,kernel_size=(1, 1), kernel_init=self.kernel_init, bias_init=self.bias_init)(z5_dropout)
        return z1, z2, z3, z4_dropout, z5_dropout


class Decoder(nn.Module):
    features: int = 128
    out: int = 1
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, z1, z2, z3, z4_dropout, z5_dropout):
        z6_up = jax.image.resize(z5_dropout, shape=(z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2, z5_dropout.shape[3]),
                                 method='nearest')
        z6 = nn.Conv(self.features * 8, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6_up)
        z6 = nn.relu(z6)
        z6 = jnp.concatenate([z4_dropout, z6], axis=3)
        z6 = nn.Conv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6)
        z6 = nn.relu(z6)
        z6 = nn.Conv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6)
        z6 = nn.relu(z6)

        z7_up = jax.image.resize(z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
                                 method='nearest')
        z7 = nn.Conv(self.features * 4, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7_up)
        z7 = nn.relu(z7)
        z7 = jnp.concatenate([z3, z7], axis=3)
        z7 = nn.Conv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7)
        z7 = nn.relu(z7)
        z7 = nn.Conv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7)
        z7 = nn.relu(z7)

        z8_up = jax.image.resize(z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
                                 method='nearest')
        z8 = nn.Conv(self.features * 2, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8_up)
        z8 = nn.relu(z8)
        z8 = jnp.concatenate([z2, z8], axis=3)
        z8 = nn.Conv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8)
        z8 = nn.relu(z8)
        z8 = nn.Conv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8)
        z8 = nn.relu(z8)

        z9_up = jax.image.resize(z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
                                 method='nearest')
        z9 = nn.Conv(self.features, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9_up)
        z9 = nn.relu(z9)
        z9 = jnp.concatenate([z1, z9], axis=3)
        z9 = nn.Conv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9)
        z9 = nn.relu(z9)
        z9 = nn.Conv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9)
        z9 = nn.relu(z9)
        y = nn.Conv(self.out, kernel_size=(1, 1), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9)
        # y = nn.ConvTranspose(self.out,kernel_size=(4,4),strides=(2,2),padding=((2,2),(2,2)),kernel_init=self.kernel_init,use_bias=False)(z9)
        return y


class UNet(nn.Module):
    features: int = 64
    out: int = 1

    @nn.compact
    def __call__(self, x, training):
        x = jnp.transpose(x, (0, 2, 3, 1))
        z1, z2, z3, z4_dropout, z5_dropout = Encoder(self.features)(x, training)
        y = Decoder(self.features, out=self.out)(z1, z2, z3, z4_dropout, z5_dropout, training)
        y = jnp.transpose(y, (0, 3, 1, 2))
        return y