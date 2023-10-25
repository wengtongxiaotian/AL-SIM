#231014
import numpy as np
import jax.numpy as jnp
import sys
sys.path.extend('.')
import tool
import cv2
def distort(img, factor):
    h, w = img.shape[:2]
    y, x = np.indices((h, w))
    x = (x - w/2) / (w/2)
    y = (y - h/2) / (h/2)
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)
    # theta = theta * factor
    # r = r * r
    r = factor(r)
    # return r
    x = r * np.cos(theta) * w/2 + w/2
    y = r * np.sin(theta) * h/2 + h/2
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)

def twist(I):
    fish = []
    for ii in range(I.shape[0]):
        for jj in range(9):
            # Define a list of non-linear functions
            functions = [
                np.sin, np.cos, np.tan, np.arctan,
                np.sinh, np.cosh, np.tanh, np.arcsinh,
                #np.exp, #np.log, np.log2, np.log10,
                lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4
            ]
            # Randomly select three non-linear functions
            func1, func2, func3 = np.random.choice(functions, size=3, replace=False)
            # Randomly select three coefficients that sum up to 1
            coeffs = np.random.dirichlet((1, 1, 1))
            # Combine the functions with the coefficients to create a new function
            new_func = lambda x: coeffs[0]*func1(x) + coeffs[1]*func2(x) + coeffs[2]*func3(x)
            # new_func = functions[i]
            indexi, indexj = np.random.randint(0,128,(2))
            fishpic = np.array(I[ii,jj])
            fish.append(distort(fishpic, new_func)[indexi:indexi+256,indexj:indexj+256])#
    # check_shape(*fish)
    I = jnp.stack(fish).reshape(-1,9,256,256)
    return I


        # coffpath = np.random.choice(os.listdir('/dataf/gpt/_record/newdataset/terrain'))
        # coffpath = os.path.join('/dataf/gpt/_record/newdataset/terrain',coffpath)
        # arr = skimread(coffpath)
        # I = I*arr[np.random.choice(1000, cfg['bs']*9),128:128+256,128:128+256].reshape(-1,9,256,256)

if __name__ == '__main__':

    # I = xx({'bs':2,'cropsize':256})
    # tool.savetif(I,tool.running_path+'/I.tif')
    I = tool.tiff.imread(tool.running_path+'/I.tif')
    print(tool.tikcount())
    distort_I = twist(I)
    check_shape(I,distort_I)
    tool.savepng(merge_display(I,3,3),tool.running_path+'/0.png')
    tool.savepng(merge_display(distort_I,3,3),tool.running_path+'/1.png')
    # tool.savetif(emitter,tool.create_path('tif'))
    # tool.savetif(raw,tool.create_path('tif'))
    # tool.savetif(S,tool.create_path('tif'))
    # tool.savetif(bg,tool.create_path('tif'))
    # tool.savetif(tau_e,tool.create_path('tif'))
    tool.exit()

