
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import numpy as np
from scipy.signal import fftconvolve
from skimage import io
from skimage.transform import resize
from scipy.interpolate import make_interp_spline
from utils_imageJ import save_tiff_imagej_compatible
import tqdm
import jax.numpy as jnp
import jax
from jax import vmap
from utils_jax import jax_jv
from scipy.special import comb
import cv2
from noise import snoise2
from multiprocessing import Pool
import pdb

params = {
        "lambda": [500, 1000],
        "NA": 0.9,
        "stepxy": 80,
        "stepz": 200,
        "save_dir": "/dataf/b/data/simulation_pts",
        "sizexy": 256,
        "PSF_sizexy": 21,
        "lines_num": 10,
        "pt_num": 1000,
        "train_num": 1000,
        "test_num": 100,
        "time": 194,
        "noise_scale": 0.5,
    }




def debye_diffraction(x, y, z):
    _lambda = np.random.uniform(params["lambda"][0], params["lambda"][1])
    _NA = params["NA"]
    k = 2 * jnp.pi / _lambda
    rho = jnp.sqrt(x**2 + y**2)
    linspace_theta = jnp.linspace(0, jnp.arcsin(_NA), 64)

    def _f(theta):
        return jnp.sqrt(jnp.cos(theta)) * jax_jv(0, k*rho*jnp.sin(theta)) * \
               jnp.exp(-1j*k*z*jnp.cos(theta)) * jnp.sin(theta)

    h = integrate1d(linspace_theta, _f)
    return h


def integrate1d(linspace_x, f):
    return jnp.trapz(f(linspace_x), x=linspace_x, axis=0)


def standard_psf_debye(sizexy, sizez):
    
    linspace_x = jnp.linspace(-(sizexy-1)/2*params["stepxy"], (sizexy-1)/2*params["stepxy"], sizexy)
    linspace_y = jnp.linspace(-(sizexy-1)/2*params["stepxy"], (sizexy-1)/2*params["stepxy"], sizexy)
    linspace_z = jnp.linspace(-(sizez-1)/2*params["stepz"], (sizez-1)/2*params["stepz"], sizez)
    xx, yy, zz = jnp.meshgrid(linspace_x, linspace_y, linspace_z)

    def _debye_diffraction(x, y, z):
        return debye_diffraction(x, y, z)

    h = vmap(_debye_diffraction, in_axes=(0, 0, 0), out_axes=(0))(xx.flatten(), yy.flatten(), zz.flatten())
    hout = jnp.abs(h).reshape(xx.shape)[...]
    return jnp.square(hout)

def convolve_curve_with_psf(curve, psf):
    convolved = fftconvolve(curve, psf, mode='same')
    return convolved


def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(points, num_points=100):
    n = len(points) - 1
    curve = np.zeros((num_points, 2))
    for i in range(num_points):
        t = i / float(num_points - 1)
        for j in range(len(points)):
            curve[i] += bernstein_poly(j, n, t) * points[j]
    return curve

def generate_bezier_curves_2d():
    m = params["lines_num"]
    n = 4 * params["sizexy"]
    img = np.zeros((n, n))#先上采样四倍
    for _ in range(m):
        num_points = np.random.randint(3, 6)
        control_points = np.random.rand(num_points, 2) * n
        curve_points = bezier_curve(control_points, num_points=n)
        prev_point = None
        for point in curve_points:
            x, y = int(point[0]), int(point[1])
            if prev_point is not None:
                px, py = prev_point
                line_points = list(zip(np.linspace(py, y, max(abs(y-py), 1)), np.linspace(px, x, max(abs(x-px), 1))))
                for lp_y, lp_x in line_points:
                    lx, ly = int(lp_x), int(lp_y)
                    if 0 <= lx < n and 0 <= ly < n:
                        img[ly, lx] = np.random.randint(160, 255)
            prev_point = (x, y)
    img = resize(img,(n//4,n//4))
    img = img/255
    img = cv2.GaussianBlur(img.astype(np.float32),(5, 5),0)
    return img

def generate_pts_2d():
    n = params["sizexy"]
    img = np.zeros((n, n))#先上采样四倍

    # Generate all the points at once
    pt_num = np.random.randint(params["pt_num"]//10, params["pt_num"])
    xs, ys = np.random.randint(0, n, size=(2, pt_num))
    
    # Ensure no overlapping indices (may reduce the number of points)
    unique_indices = np.unique((xs, ys), axis=1)
    xs, ys = unique_indices

    # Assign random intensity values to each point
    # img[ys, xs] = np.random.randint(255, 255, size=xs.shape)
    img[ys, xs] = 255
    # img = resize(img,(n//4,n//4))
    img = img / 255
    # img = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
    return img


def wave_like_mesh(r=0.05):
    # 定义网格大小
    size = params["sizexy"]

    # 创建一个存储噪声值的二维数组
    waves = np.zeros((size, size))

    # 定义噪声的频率
    freq = r * np.random.rand()

    # 定义每个图像的随机偏移量
    offsets = np.random.rand(2,) * 1000  # 随机偏移量范围为[0, 1000]

    # 创建网格
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)

    # 遍历每一个子图
    for i in range(size):
        for j in range(size):
            waves[i][j] = snoise2(i * freq + offsets[0], j * freq + offsets[1])
    return waves

def process(num):
    np.random.seed(num)
    psf = standard_psf_debye(params["PSF_sizexy"], 1).transpose()
    I = []
    for i in range(3):
        coco_item = coco[num*3+i]
        coco_item -= coco_item.min()
        coco_item /= coco_item.max()
        phi = coco_item*np.pi*2
        # I.append([])
        I1 = []
        phi0 = np.random.rand()*np.pi
        for j in range(3):
            I1.append(np.sin(phi+j*np.pi/3*2+phi0)+1)
        I1 = np.stack(I1)
        I1 = I1/I1.mean(axis=0,keepdims=True)
        I.append(I1)
    I = np.concatenate(I)
    save_tiff_imagej_compatible(f'{train_set_gt_path}/{num}_lf.tif', I.astype(np.float32), "CYX")
    return 0
    psf /= psf.sum()
    pts = generate_pts_2d()
    emitter = pts * (wave_like_mesh(0.1) + 1)
    tau = 10 * pts * wave_like_mesh(0.1) + 20 + 10 * np.random.rand() + 10 * wave_like_mesh(0.001) # 7 - 13
    t0 = wave_like_mesh(0.5) * 0.1 + 5 + 40 * np.random.rand() # 1 - 12
    ts = np.arange(0, params["time"], 1)[:, None, None]
    ts = np.tile(ts, (1, params["sizexy"], params["sizexy"]))
    bg = (wave_like_mesh(0.001)) * np.random.rand()
    bg -= bg.min()
    S = (emitter + bg) * np.exp(-(ts - t0) / tau) * 1 / 2 * (1 + np.tanh(100 * (ts - t0)))
    raw = convolve_curve_with_psf(S, psf)
    noise = raw.mean() * np.random.normal(loc=0.0, scale=np.random.uniform(0.1, params["noise_scale"]), size=raw.shape)
    raw = raw + noise
    if i < params["test_num"]:
        gt_path = test_set_gt_path
        set_path = test_set_path
    else:
        gt_path = train_set_gt_path
        set_path = train_set_path
    save_tiff_imagej_compatible(f'{gt_path}/{i}_tau.tif', tau.astype(np.float32), "YX")
    save_tiff_imagej_compatible(f'{gt_path}/{i}_t0.tif',  t0.astype(np.float32), "YX")
    save_tiff_imagej_compatible(f'{gt_path}/{i}_emitter.tif',  emitter.astype(np.float32), "YX")
    save_tiff_imagej_compatible(f'{gt_path}/{i}_bg.tif',  bg.astype(np.float32), "YX")
    save_tiff_imagej_compatible(f'{gt_path}/{i}_psf.tif',  psf[0, ...].astype(np.float32), "YX")
    save_tiff_imagej_compatible(f'{set_path}/{i}.tif', raw.astype(np.float32), "TYX")
    
    

    

if __name__ == "__main__":
    
    save_dir = params["save_dir"]
    # Saving path
    # train_set_path = os.path.join(save_dir, 'train')
    # os.makedirs(train_set_path, exist_ok=True)
    train_set_gt_path = os.path.join(save_dir, 'train_gt')
    os.makedirs(train_set_gt_path, exist_ok=True)
    # test_set_path = os.path.join(save_dir, 'test')
    # test_set_gt_path = os.path.join(save_dir, 'test_gt')
    # os.makedirs(test_set_path, exist_ok=True)
    # os.makedirs(test_set_gt_path, exist_ok=True)


    num_list = range(params["train_num"] + params["test_num"])
    # for i in tqdm.tqdm(num_list):
    #     process(i)
    import skimage
    coco = skimage.io.imread('/dataf/ndl/_record/newdataset/coco_bg.tif')
    coco = (coco-coco.min())/(coco.max()-coco.min())
    with Pool(processes=32) as pool:
        max_ = len(num_list)
        with tqdm.tqdm(total=max_) as pbar:
            for i, _ in tqdm.tqdm(enumerate(pool.imap_unordered(process, num_list))):
                pbar.update()