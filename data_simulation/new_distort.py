#pip install noise #效果不好
import numpy as np
import os
import jax
import sys
import jax.numpy as jnp
sys.path.extend('.')
from noise import snoise2
import tool
params = {
        "sizexy": 256,
    }
def wave_like_mesh(r=0.05,size=256,):
    # size定义网格大小

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
    return waves - waves.min()
def check_shape(*arrays):
    shapelist = [arr.shape if hasattr(arr,'shape') else len(arr) for arr in arrays]
    print(shapelist,tool.printinfo(2))
def merge_display(array,width,height): #并排显示图像
    hwc = array.shape[-3:] if array.shape[-1] == 3 else array.shape[-2:]
    array = array.reshape(-1,*hwc)
    array = np.concatenate([array,np.zeros((width*height,*hwc))]) #array填不满图像时，用零填充
    array = np.swapaxes(array[:width*height].reshape(height,width,*hwc),1,2)
    return array.reshape(height*hwc[0],width*hwc[1],-1).squeeze()
if __name__ == '__main__':
    # I = xx({'bs':2,'cropsize':256})
    # tool.savetif(I,tool.running_path+'/I.tif')
    print(tool.tikcount())
    # print(I.shape)
    # tool.savepng(I,tool.create_path('png'))
    # size = 256
    # noise = wave_like_mesh(size)
    # bg = (wave_like_mesh(size,0.01)*0.01 + 1) * np.random.rand()
    # print(noise.shape)
    # tool.savepng(bg,tool.create_path('png'))

    emitter = (wave_like_mesh(0.1) + 1) * np.random.rand()
    # tau = 30 + 3 * wave_like_mesh(0.01) - (emitter / emitter.max()) * wave_like_mesh(0.01) * 10 # 7 - 13
    # tau_e = (emitter_t / emitter_t.max()) * wave_like_mesh(0.003)
    # tau_e = tau_e / tau_e.max()
    # tau = 30 + 3 * wave_like_mesh(0.01) -  tau_e * 15 # 7 - 13
    t0 = wave_like_mesh(0.5) * 0.1 + 40 + 10 * np.random.rand() # 1 - 12
    ts = np.arange(0, 9, 1)[:, None, None]
    ts = np.tile(ts, (1, 256, 256))
    bg = (wave_like_mesh(0.01)*0.01 + 1) * np.random.rand()
    # S = emitter * jnp.exp(-(ts - t0) / tau) * 1 / 2 * (1 + jnp.tanh(100 * (ts - t0)))
    # raw = S + bg

    # raw_min = raw.min()
    # raw_std = raw.std()
    # raw = (raw - raw_min) / raw_std
    # emitter = emitter / raw_std
    # bg = (bg - raw_min) / raw_std
    # raw = merge_display(raw[0], 3,3)
    # check_shape(emitter, raw, S, bg, tau_e)
    import os
    tool.savetif(ts,tool.create_path(tool.exec_dir,'0.tif'))
    # tool.savepng(merge_display(tau_e[0],3,3),os.path.join(tool.exec_dir,'1.png'))
    # tool.savetif(emitter,tool.create_path('tif'))
    # tool.savetif(raw,tool.create_path('tif'))
    # tool.savetif(S,tool.create_path('tif'))
    # tool.savetif(bg,tool.create_path('tif'))
    # tool.savetif(tau_e,tool.create_path('tif'))

