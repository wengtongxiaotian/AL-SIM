import jax
import jax.numpy as jnp
from jax import vmap
from .utils_jax import jax_jv


def standard_psf(size_x):
    _lambda = 532
    _NA = 0.7
    _step = 50
    linspace_x = jnp.linspace(-size_x/2*_step, size_x/2*_step, size_x)
    linspace_y = jnp.linspace(-size_x/2*_step, size_x/2*_step, size_x)
    linspace_z = jnp.linspace(0, 0, 1)
    Eout_x, Eout_y, Eout_z, phase_x, phase_y, phase_z = Radially_incident_light_field(_lambda, _NA, linspace_x, linspace_y, linspace_z)
    Eout = jnp.stack([Eout_x, Eout_y], axis=0)
    psf = jnp.sum(jnp.abs(Eout)**2, axis=0)
    psf = psf / psf.sum()
    return psf[jnp.newaxis, jnp.newaxis, ..., 0], Eout[jnp.newaxis, ..., 0]
def standard_psf_debye(size,step=26):
    params = {}
    # params["lambda"] = 488 / 1.5
    # params["NA"] = 1.3 / 1.5
    # params["step"] = 62.6/2
    params["lambda"] = 700
    params["NA"] = 1.3
    params["step"] = step# 40
    linspace_x = jnp.linspace(-(size-1)/2*params["step"], (size-1)/2*params["step"], size)
    linspace_y = jnp.linspace(-(size-1)/2*params["step"], (size-1)/2*params["step"], size)
    linspace_z = jnp.linspace(0, 0, 1)
    xx, yy, zz = jnp.meshgrid(linspace_x, linspace_y, linspace_z)
    def _debye_diffraction(x, y, z):
        return debye_diffraction(x, y, z, params)
    h = vmap(_debye_diffraction, in_axes=(0, 0, 0), out_axes=(0))(xx.flatten(), yy.flatten(), zz.flatten())
    hout = jnp.square(h).reshape(xx.shape)[jnp.newaxis, jnp.newaxis, ..., 0]
    return hout/hout.sum(axis=(-1,-2))
def standard_psf_debye1(size):
    params = {}
    params["lambda"] = 700 / 1.5
    params["NA"] = 1.3 / 1.5
    params["step"] = 40
    linspace_x = jnp.linspace(-(size-1)/2*params["step"], (size-1)/2*params["step"], size)
    linspace_y = jnp.linspace(-(size-1)/2*params["step"], (size-1)/2*params["step"], size)
    linspace_z = jnp.linspace(0, 0, 1)
    zz, yy, xx = jnp.meshgrid(linspace_z, linspace_y, linspace_x, indexing='ij')
    def _debye_diffraction(x, y, z):
        return debye_diffraction(x, y, z, params)
    h = vmap(_debye_diffraction, in_axes=(0, 0, 0), out_axes=(0))(xx.flatten(), yy.flatten(), zz.flatten())
    hout = jnp.square(h).reshape(xx.shape)[jnp.newaxis, jnp.newaxis, ..., 0]
    return hout/hout.sum(axis=(-1,-2))


def debye_diffraction(x, y, z, params):
    _lambda = params["lambda"]
    _NA = params["NA"]
    k = 2 * jnp.pi / _lambda
    rho = jnp.sqrt(x**2 + y**2)
    linspace_theta = jnp.linspace(0, jnp.arcsin(_NA), 64)
    def _f(theta):
        return jnp.sqrt(jnp.cos(theta)) * jax_jv(0, k*rho*jnp.sin(theta)) * \
                    jnp.exp(-k*z*jnp.cos(theta)) * jnp.sin(theta)
    h = integrate1d(linspace_theta, _f)
    return h


def modified_psf_asr(size, vf):
    # vf : [1, n_frames, n_alpha, n_theta]
    params = {}
    params["M"] = 40
    params["lambda"] = 700 / 1.5
    params["NA"] = 1.3 / 1.5 / params["M"]
    params["step"] = 40 * params["M"]
    linspace_x = jnp.linspace(-(size-1)/2*params["step"], (size-1)/2*params["step"], size)
    linspace_y = jnp.linspace(-(size-1)/2*params["step"], (size-1)/2*params["step"], size)
    linspace_z = jnp.linspace(0, 0, 1)
    linspace_theta = jnp.linspace(0, jnp.arcsin(params["NA"]), vf.shape[-1])
    linspace_alpha = jnp.linspace(0, 2*jnp.pi, vf.shape[-2])
    Eout_x, Eout_y, Eout_z, _, _, _ = tight_focus(vf, jnp.zeros_like(vf), linspace_theta, linspace_alpha, linspace_x, linspace_y, linspace_z, params["lambda"])
    psf = jnp.square(jnp.abs(Eout_x)) + jnp.square(jnp.abs(Eout_y)) + jnp.square(jnp.abs(Eout_z))
    psf = psf / psf.sum(axis=(-1, -2), keepdims=True)
    return psf


def integrate1d(linspace_x, f):
    """
    Integrate a 1D function over a grid.
    """
    return jnp.trapz(f(linspace_x), x=linspace_x, axis=0)


def integrate2d(linspace_x, linspace_y, f):
    """
    Integrate a 2D function over a grid.
    """
    Y, X = jnp.meshgrid(linspace_y, linspace_x, indexing='ij')
    X = X[jnp.newaxis, jnp.newaxis, ...]
    Y = Y[jnp.newaxis, jnp.newaxis, ...]
    return jnp.trapz(jnp.trapz(f(X, Y), x=linspace_x, axis=-1), x=linspace_y, axis=-1)

def asr_of_focal_field(linspace_theta, linspace_alpha, Ex, Ey, _k, _rho, _phi, _z):
    """
        Compute the asr of a focal field.

        Ex: [1, n_frames, n_alpha 4, n_theta]
        Ey: [1, n_frames, n_alpha 4, n_theta]
    """
    def angular_spectrum_representation(E_in, THETA, ALPHA):
        return E_in * jnp.exp(1j*_k*_z*jnp.cos(THETA)) * \
                jnp.exp(1j*_k*_rho*jnp.sin(THETA)*jnp.cos(ALPHA-_phi)) * jnp.sin(THETA)
    
    def f_Ex(theta, alpha):
        E_in = ( -1 * ( jnp.cos( theta ) )**( 1/2 ) * jnp.sin( alpha ) * ( Ey * \
                jnp.cos( alpha ) + -1 * Ex * jnp.sin( alpha ) ) + jnp.cos( \
                alpha ) * ( jnp.cos( theta ) )**( 3/2 ) * ( Ex * jnp.cos( alpha ) \
                + Ey * jnp.sin( alpha ) ) )
        return angular_spectrum_representation(E_in, theta, alpha)

    def f_Ey(theta, alpha):
        E_in = ( jnp.cos( alpha ) * ( jnp.cos( theta ) )**( 1/2 ) * ( Ey * \
                jnp.cos( alpha ) + -1 * Ex * jnp.sin( alpha ) ) + ( jnp.cos( \
                theta ) )**( 3/2 ) * jnp.sin( alpha ) * ( Ex * jnp.cos( alpha ) + \
                Ey * jnp.sin( alpha ) ) )
        return angular_spectrum_representation(E_in, theta, alpha)

    def f_Ez(theta, alpha):
        E_in = -1 * ( jnp.cos( theta ) )**( 1/2 ) * ( Ex * jnp.cos( alpha ) + Ey \
                * jnp.sin( alpha ) ) * jnp.sin( theta )
        return angular_spectrum_representation(E_in, theta, alpha)

    Eout_x = integrate2d(linspace_theta, linspace_alpha, f_Ex)
    Eout_y = integrate2d(linspace_theta, linspace_alpha, f_Ey)
    Eout_z = integrate2d(linspace_theta, linspace_alpha, f_Ez)
    return Eout_x, Eout_y, Eout_z


def tight_focus(Ex, Ey, linspace_theta, linspace_alpha, linspace_x, linspace_y, linspace_z, _lambda):
    _k = 2 * jnp.pi / _lambda
    Z, Y, X = jnp.meshgrid(linspace_z, linspace_y, linspace_x, indexing='ij')
    _rho = jnp.sqrt(X**2 + Y**2).flatten()
    _phi = jnp.angle(X + Y * 1j).flatten()
    _z = Z.flatten()
    def _asr_of_focal_field(_rho, _phi, _z):
        return asr_of_focal_field(linspace_theta, linspace_alpha, Ex, Ey, _k, _rho, _phi, _z)
    Eout_x, Eout_y, Eout_z = vmap(_asr_of_focal_field, in_axes=(0, 0, 0), out_axes=(2, 2, 2))(_rho, _phi, _z)
    out_size = [Ex.shape[0], Ex.shape[1], X.shape[1], X.shape[2]]
    phase_x, phase_y, phase_z = jnp.angle(Eout_x).reshape(out_size), jnp.angle(Eout_y).reshape(out_size), jnp.angle(Eout_z).reshape(out_size)
    Eout_x = jnp.array(Eout_x).reshape(out_size)
    Eout_y = jnp.array(Eout_y).reshape(out_size)
    Eout_z = jnp.array(Eout_z).reshape(out_size)
    return Eout_x, Eout_y, Eout_z, phase_x, phase_y, phase_z


def fft_fraunhofer(x):
    x = jnp.fft.fftshift(x)
    x = jnp.fft.fft2(x)
    x = jnp.fft.fftshift(x)
    return x

def ifft_fraunhofer(x):
    x = jnp.fft.ifftshift(x)
    x = jnp.fft.ifft2(x)
    x = jnp.fft.ifftshift(x)
    return x

def rfft_fraunhofer(x):
    x = jnp.fft.rfft2(x)
    x = jnp.fft.fftshift(x)
    return x

def irfft_fraunhofer(x):
    x = jnp.fft.ifftshift(x)
    x = jnp.fft.irfft2(x)
    return x


def fft_drift(Ein, dkx=0, dky=0):
    h, w = Ein.shape[-2], Ein.shape[-1]
    yy, xx = jnp.meshgrid(jnp.arange(h)-(h-1)/2, jnp.arange(w)-(w-1)/2, indexing='ij')
    Ein = Ein*jnp.exp(-1j*2*jnp.pi*dkx*xx/w)*jnp.exp(-1j*2*jnp.pi*dky*yy/h)
    return Ein


def gaussian_2d(X, Y, mu_x, mu_y, sigma_x, sigma_y):
    return jnp.exp(-(X - mu_x) ** 2 / (2 * sigma_x ** 2) - (Y - mu_y) ** 2 / (2 * sigma_y ** 2))


def make_mask(x, dr):
    # mask
    vy = jnp.linspace(-0.5, 0.5, x.shape[2]) * x.shape[2]
    vx = jnp.linspace(-0.5, 0.5, x.shape[3]) * x.shape[3]
    yy, xx = jnp.meshgrid(vy, vx, indexing='ij')
    rr = jnp.sqrt(xx**2 + yy**2)
    mask = jax.lax.stop_gradient(jnp.where(rr <= dr, 1, 0)[jnp.newaxis, jnp.newaxis, ...])
    return mask

def unmask(img, x, y, dr, value):
    # mask
    vy = jnp.linspace(-1, 1, img.shape[2]) * img.shape[2]
    vx = jnp.linspace(-1, 1, img.shape[3]) * img.shape[3]
    yy, xx = jnp.meshgrid(vy, vx, indexing='ij')
    rr = jnp.sqrt((xx-x)**2 + (yy-y)**2)
    rr = rr[jnp.newaxis, jnp.newaxis, ...]
    img = img.at[jnp.where(rr <= dr)].set(value)
    return img

def make_gaussian(x, dr):
    # mask
    vy = jnp.linspace(-1, 1, x.shape[2]) * x.shape[2]
    vx = jnp.linspace(-1, 1, x.shape[3]) * x.shape[3]
    yy, xx = jnp.meshgrid(vy, vx, indexing='ij')
    rr = jnp.sqrt(xx**2 + yy**2)
    mask = jax.lax.stop_gradient(gaussian_2d(xx, yy, 0, 0, dr, dr)[jnp.newaxis, jnp.newaxis, ...])
    return mask


def pattern_generation(M, Ks, n_angle, n_phase, phi, crop_size):
    kx, ky = Ks[...,0],Ks[...,1]
    I = []
    for a in range(n_angle):
        for p in range(n_phase):
            xs = jnp.linspace(0, crop_size - 1, crop_size)
            ys = jnp.linspace(0, crop_size - 1, crop_size)
            xx, yy = jnp.meshgrid(xs, ys, indexing='ij')
            I.append(1 + M[a]*jnp.cos(2*jnp.pi*(kx[a]*xx/crop_size+ky[a]*yy/crop_size)+ phi[a] + 2/3*p*jnp.pi))
    I = jnp.stack(I, axis=0)[jnp.newaxis, ...]
    I = I / jnp.mean(I, axis=(2, 3), keepdims=True)
    return I

def get_Ks(theta_start, n_angle, rho):
    start_k = jnp.array([jnp.cos(theta_start), jnp.sin(theta_start)])
    ks = jnp.stack([rot2d(start_k, i*2*jnp.pi/n_angle) for i in range(n_angle)])[..., 0] * rho # [n_angle, 2]
    return ks

def rot2d(A, theta):
    rot = jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])
    return jnp.matmul(rot, A)
def pattern_generation_jax(cfg,key_new,):
    # cfg["cropsize"] = 2*cfg["cropsize"]
    # key_new = state.rng
    SIM_params = {
            "M": [0.2, 0.8], #[0.5, 0.8],
            "theta_start": [0, 2*jnp.pi], #[0, 0],
            "n_angle": 3,
            "n_phase": 3,
            "phi": [0, 2*jnp.pi],
            "rho": [150 / 2048,  250 / 2048], #[args.f_th / si_size,  args.f_th / si_size]
        }
    n_angle = SIM_params["n_angle"]
    n_phase = SIM_params["n_phase"]
    M = jax.random.uniform(key_new, shape=(cfg['bs'],SIM_params["n_angle"],), minval=SIM_params["M"][0], maxval=SIM_params["M"][1])
    theta_start = jax.random.uniform(key_new, shape=(cfg['bs'],1,), minval=SIM_params["theta_start"][0], maxval=SIM_params["theta_start"][1])
    phi = jax.random.uniform(key_new, shape=(cfg['bs'],SIM_params["n_angle"],), minval=SIM_params["phi"][0], maxval=SIM_params["phi"][1])
    rho = jax.random.uniform(key_new, shape=(cfg['bs'],SIM_params["n_angle"],1), minval=SIM_params["rho"][0], maxval=SIM_params["rho"][1])
    Ks = jax.vmap(get_Ks,in_axes=(0,None,0))(theta_start, SIM_params["n_angle"], 2*rho*cfg['cropsize'])
    I = jax.vmap(pattern_generation,in_axes=(0,0,None,None,0,None))(M, Ks, n_angle, n_phase, phi, 2*cfg['cropsize'])
    return I[:,0]

if __name__ == "__main__":
    
    from matplotlib import pyplot as plt
    import cv2
    import numpy as np
    import pdb

    in_size = 128
    out_size = 128
    mask_r = 1000


    psf_i = standard_psf_debye(in_size)
    psf_i = psf_i / psf_i.max()

    
    E_out = fft_fraunhofer(psf_i)
    
    mask = make_mask(E_out, mask_r)
    E_r1 = ifft_fraunhofer(fft_drift(E_out, 10, 10))
    E_r2 = ifft_fraunhofer(fft_drift(E_out, -10, -0))

    psf_o = jnp.sum(jnp.square(jnp.abs(E_out)), axis=1, keepdims=True)
    psf_o = psf_o / psf_o.max()

    psf_r1 = jnp.sum(jnp.square(jnp.abs(E_r1)), axis=1, keepdims=True)
    psf_r1 = psf_r1 / psf_r1.max()

    psf_r2 = jnp.sum(jnp.square(jnp.abs(E_r2)), axis=1, keepdims=True)
    psf_r2 = psf_r2 / psf_r2.max()

    plt.subplot(141)
    plt.imshow(np.abs(psf_i[0, 0, ...]))

    plt.subplot(142)
    plt.imshow(np.abs(psf_o[0, 0, ...]))

    plt.subplot(143)
    plt.imshow(np.abs(psf_r1[0, 0, ...]))

    plt.subplot(144)
    plt.imshow(np.abs(psf_r2[0, 0, ...]))

    plt.show()


    # def two_beam(im, dx, dr):
    #     center_x = im.shape[0] // 2 
    #     center_y = im.shape[1] // 2
    #     im = cv2.circle(im, (center_x-int(dx), center_y), int(dr/2), 1, -1)
    #     im = cv2.circle(im, (center_x+int(dx), center_y), int(dr/2), 1, -1)
    #     return im

    # def pattern(Ein, dkx=0, dky=0):
    #     Ein = fft_drift(Ein, dkx, dky)
    #     Eout = np.array(fft_fraunhofer(Ein))
    #     Iout = np.sum(np.abs(Eout)**2, axis=(0, 1))
    #     Iout = Iout / Iout.max() * 255
    #     return Iout


    # params = {}
    # params["Ein_size"] = 512

    # Ein = np.zeros([1, 1, params["Ein_size"], params["Ein_size"]]) 
    # Ein[0, 0, ...] = two_beam(Ein[0, 0, ...], 5, 1)
    # Iin = np.sum(np.abs(Ein)**2, axis=(0, 1))
    # Iin = Iin / Iin.max() * 255
    
    # plt.rcParams["figure.figsize"] = (15, 6)

    # plt.subplot(141)
    # sz = params["Ein_size"] // 2
    # plt.imshow(Iin)
    # plt.title('Input intensity')

    # plt.subplot(142)
    # Iout = pattern(Ein, dkx=0, dky=0)
    # plt.imshow(Iout)
    # plt.title('Output intensity')

    # plt.subplot(143)
    # Iout = pattern(Ein, dkx=33, dky=0)
    # plt.imshow(Iout)
    # plt.title('Output intensity')

    # plt.subplot(144)
    # Iout = pattern(Ein, dkx=66, dky=0)
    # plt.imshow(Iout)
    # plt.title('Output intensity')

    # plt.show()

    
    
    

