import jax.numpy as jnp
import scipy.special
from jax import custom_jvp, pure_callback, vmap
import numpy as np

# see https://github.com/google/jax/issues/11002


def generate_bessel(function):
    """function is Jv, Yv, Hv_1,Hv_2"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: -cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) - cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv




def generate_modified_bessel(function, sign):
    """function is Kv and Iv"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: sign * cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) + cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv


def spherical_bessel_genearator(f):
    def g(v, x):
        return f(v + 0.5, x) * jnp.sqrt(jnp.pi / (2 * x))

    return g


def jax_jv(v, x):
    jv = generate_bessel(scipy.special.jv)
    y = vmap(jv, in_axes=(None, 0))(v, x.flatten())
    return y.reshape(x.shape)

if __name__ == "__main__":
    jv = generate_bessel(scipy.special.jv)
    yv = generate_bessel(scipy.special.yv)
    kv = generate_modified_bessel(scipy.special.kv, sign=-1)
    iv = generate_modified_bessel(scipy.special.iv, sign=+1)
    hankel1 = generate_bessel(scipy.special.hankel1)
    hankel2 = generate_bessel(scipy.special.hankel2)
    spherical_jv = spherical_bessel_genearator(jv)
    spherical_yv = spherical_bessel_genearator(yv)
    spherical_hankel1 = spherical_bessel_genearator(hankel1)
    spherical_hankel2 = spherical_bessel_genearator(hankel2)
    
    x = jnp.linspace(0.0, 20.0, num=1000)
    y = jax_jv(0, x)
    print(y.shape)

    # import matplotlib.pyplot as plt

    # x = jnp.linspace(0.0, 20.0, num=1000)


    # for func, name in zip(
    #     [jv],
    #     ["jv"],
    # ):

    #     plt.figure()

    #     for i in range(5):
    #         y = vmap(func, in_axes=(None, 0))(i, x)
    #         plt.plot(x, y, label=i)

    #     plt.ylim([-1.1, 1.1])
    #     plt.title(name)
    #     plt.legend()

    #     plt.draw()
    #     # plt.pause(0.001)

    #     plt.show()

    # print("done")