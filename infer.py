#get_pth get_data get_cfg get_model get_result get_score
import sys
sys.path.append(".")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['expn'] = 'alsim-infer'
from tools.tool import *
import jax.numpy as jnp
import jax
from flax.training import train_state, checkpoints
from typing import Any
from tools.pic import stitch_images,split_image
from flax import struct
@struct.dataclass
class TrainState:
    params: Any

traincfg = 'bs:2 features:128 cropsize:256 debug:false'
modcfg(traincfg)
modcfg('psfsize:27 scale:1')
exec_dir = '/dataf/b/_record/pointandline1220/hs'
ckptdir = exec_dir+'/ckpt'
testdata = tiff.imread(f'/dataf/b/data/SIM00051.tif')
from network import DND_SIM
rng = jax.random.PRNGKey(42)
net = DND_SIM(cfg['features'])
apply_fn=net.apply # y_pred = apply_fn({'params': state.params}, x)
state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckptdir, target=None)
state = TrainState(params=state_dict['params'],)

print(cfg)
# def test(): #test
def norm2(arr): #(bs,h,w)
    mean = arr.mean()
    var = arr.var()
    print(mean,var)
    return (arr-mean)/(var**.5)
def norm_train(arr): #(bs,h,w)
    mean = arr.mean(axis=(-4,-3,-2,-1),keepdims=True)
    var = arr.var(axis=(-4,-3,-2,-1),keepdims=True)
    return (arr-mean)/(var+1e-6)**.5, mean, var
def norm1(arr):
    mean = arr.mean(axis=(-2,-1),keepdims=True)
    var = arr.var(axis=(-2,-1),keepdims=True)
    return (arr-arr.min())/(var+1e-6)**.5,mean,var
def norm(arr):
    maxa = arr.max(axis=(-2,-1),keepdims=True)
    mina = arr.min(axis=(-2,-1),keepdims=True)
    return (arr-mina)/(maxa-mina)
def norm3(arr):
    maxa = arr.max(keepdims=True)
    mina = arr.min(keepdims=True)
    return (arr-mina)/(maxa-mina)
# from toolfunctions.myarray import norm_test as norm
def testwrapfn(arr3d,patchsize=cfg['cropsize'],bs=cfg['bs'],norm_fn=norm1,iteridx=0):
    arr4d = split_image(arr3d,patchsize,64)
    print(arr4d.shape)
    while iteridx < arr4d.shape[0]:
        arr = arr4d[iteridx:iteridx+bs]
        arr,mean,var = norm_fn(arr)
        mean,var = 0,1
        iteridx += bs
        yield arr,mean,var
array = norm(testdata)
# array = jax.image.resize(array,(array.shape[0],array.shape[1]*cfg['scale'],array.shape[2]*cfg['scale']),method='linear')
# array = testdata
testset = testwrapfn(array,norm_fn = norm1)
modelout,modelin = [],[]
res_p = []
# @jax.jit
# def acc(params,x):
#     res = 
#     return res
# def apply_net(params, x, rng): return net.apply({'params': params}, x, False, rngs={'dropout': rng})
jit_apply_net = jax.jit(lambda params, x, rng:net.apply({'params': params}, x, False, rngs={'dropout': rng}))
import tqdm
for x,mean,var in tqdm.tqdm(testset):
    res = jit_apply_net(state.params, x, rng)
    y = res['rec']
    modelout.append(np.array(y))
    modelin.append(np.array(x))
    res_p.append(np.array(res['rec_p']))
pdb()
test_res = np.concatenate(modelout)
test_p = np.concatenate(res_p)
patchs = array.shape[-1]//cfg['cropsize']*2
channels = 1
patchsize = cfg['cropsize']
padsize = cfg['cropsize']//4
imgsize = array.shape[-1]

test_res = stitch_images(test_res,64,16,16)
channels = 9
test_res_p = stitch_images(test_p,64,16,16)
test_res_p = jnp.concatenate([test_res_p,array])
test_res = np.concatenate([test_res,array.mean(0,keepdims=True)])
savetif(test_res,'test-minmax-norm.tif') #-nobatchnorm
savetif(test_res_p,'test55-2-pattern-norm.tif')
# savetif(np.concatenate([modelout[3],modelin[3],res_p[3]],1))
endexp()


# if __name__ == '__main__':
#     test()
