#get_pth get_data get_cfg get_model get_result get_score
import sys
sys.path.append('.')
sys.argv.append('cuda-0')
from tool import *
exec_dir = '/dataf/b/_record/cocoandcurve/ng'
ckptdir = exec_dir+'/ckpt'
#/dataf/dl/_record/pred_I_unet/zwsvx/params/121440.pkl
import pickle
import jax.numpy as jnp
# from toolfunctions.myarray import center_crop_paste
from collections import defaultdict
import jax
from flax.core.frozen_dict import freeze
from flax.training import train_state, checkpoints
import optax
def center_crop_paste(img,patchsize):
    # imgsize(c,h,h) --centercrop--> (4*p*p,c,h/p,h/p) --model--> (4*p*p,c,H/p,H/p) --centerpaste--> (c,H,H)
    #center crop
    imgsize = img.shape[-2]
    padsize = patchsize//4
    cropsize = patchsize//2
    patchnum = imgsize//cropsize
    img = jnp.pad(img,((0,0),(padsize,padsize),(padsize,padsize)))
    xx = jnp.arange(patchnum)*cropsize
    yy = jnp.arange(patchnum)*cropsize
    xs,ys = jnp.meshgrid(xx,yy)
    imglist  = []
    for i,(x1,y1) in enumerate(zip(xs.reshape(-1),ys.reshape(-1))):
        x1,y1 = int(x1),int(y1)
        imgcrop = img[:,x1:x1+2*cropsize,y1:y1+2*cropsize]
        imglist.append(imgcrop)
    modelin = jnp.stack(imglist)
    # center paste
    modelout = modelin[:,:,::2,::2]
    channels = modelout.shape[1]
    scale = modelout.shape[-2]/patchsize
    print(imgsize,patchsize,padsize,scale)
    imgsize,patchsize,padsize = int(imgsize*scale),int(patchsize*scale),int(padsize*scale),
    patchs = imgsize//patchsize*2
    res = modelout.reshape(patchs,patchs,channels,patchsize,patchsize)[:,:,:,padsize:patchsize-padsize,padsize:patchsize-padsize].transpose(2,1,3,0,4).reshape(channels,imgsize,imgsize)
    return modelin,res
traincfg = 'preiod-10_lr-0.0003_sigma-0.03_trainsize-900_validsize-90_bs-15_features-128_optim-adam_cropsize-256_mode-vscode_caption-sigma0.01_group-awareligthfield_onlypattern-1'
cfg = defaultdict(lambda:None)
cfg = update(cfg,traincfg)
# picnum = cfg['picnum']
testdata = tiff.imread(f'/dataf/b/data/SIM00051.tif')
from typing import Any
class TrainState(train_state.TrainState):
    rng : Any = None,
    batch_stats: Any = None
cfg['psfsize'] = 27
cfg['cropsize'] = 256
from network import DND_SIM
rng = jax.random.PRNGKey(42)
net = DND_SIM(cfg['features'])

state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckptdir, target=None)
state = TrainState.create(apply_fn=net.apply,
                                       params=state_dict['params'],
                                       rng = state_dict['rng'],
                                       tx=optax.sgd(0.1)   # Default optimizer
                                      )


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
    return (arr-mean)/(var+1e-6)**.5
def norm(arr):
    maxa = arr.max(axis=(-2,-1),keepdims=True)
    mina = arr.min(axis=(-2,-1),keepdims=True)
    return (arr-mina)/(maxa-mina)
def norm3(arr):
    maxa = arr.max(keepdims=True)
    mina = arr.min(keepdims=True)
    return (arr-mina)/(maxa-mina)
# from toolfunctions.myarray import norm_test as norm
def testwrapfn(arr3d,patchsize=cfg['cropsize'],bs=cfg['bs'],norm_fn=norm,iteridx=0):
    arr4d = center_crop_paste(arr3d,patchsize)[0]
    while iteridx < arr4d.shape[0]:
        arr = arr4d[iteridx:iteridx+bs]
        # arr = norm_fn(arr)
        # arr,mean,var = norm_fn(arr)
        mean,var = 0,1
        iteridx += bs
        yield arr,mean,var
array = norm(testdata)
# array = testdata
testset = testwrapfn(array,norm_fn = norm_train)
rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
modelout,modelin = [],[]
res_p = []
for x,mean,var in testset:
    res = net.apply({'params': state.params}, x , False, rngs={'dropout': rng1,})
    print(res['rec'].shape)
    y = res['rec']
    modelout.append(y)
    modelin.append(x)
    res_p.append(res['rec_p'])
test_res = jnp.concatenate(modelout)
test_p = jnp.concatenate(res_p)
patchs = 2048//256*2
channels = 1
patchsize = 256
padsize = 64
imgsize = 2048

test_res = test_res.reshape(patchs,patchs,channels,patchsize,patchsize)[:,:,:,padsize:patchsize-padsize,padsize:patchsize-padsize].transpose(2,1,3,0,4).reshape(channels,imgsize,imgsize)
channels = 9
# pdb()
test_res_p = test_p.reshape(patchs,patchs,channels,patchsize,patchsize)[:,:,:,padsize:patchsize-padsize,padsize:patchsize-padsize].transpose(2,1,3,0,4).reshape(channels,imgsize,imgsize)
# test_res_p = jnp.concatenate([test_res_p,array])
test_res = jnp.concatenate([test_res,array.mean(0,keepdims=True)])
savetif(test_res,exec_dir+'/test-minmax.tif') #-nobatchnorm
savetif(test_res_p,exec_dir+'/test55-2-pattern.tif')
savetif(jnp.concatenate([modelout[3],modelin[3],res_p[3]],1))
savetif.x


# if __name__ == '__main__':
#     test()
