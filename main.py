import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '0.9'
import sys
sys.path.append('.')
sys.path.append(os.path.dirname(__file__))
if len(sys.argv) == 1:
    sys.argv.append(f'mode-vscode_sigma-0.03_trainsize-600_validsize-90_bs-5_features-128_caption-sigma0.03')
from toolfunctions.tool import update,dict2str,create_path,tiff,glob,savetif,savepng,exec_id,printinfo,record_time
# record_time(100)
from toolfunctions.myarray import norm
from data_simulation.distort_new import twist
import jax
import time
import numpy as np
start_time = time.time()
from collections import defaultdict
from train_utils import simple_score as compute_metrics
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
from typing import Any
from ai_codes.ab_eval import Eval
cfg = defaultdict(lambda:None)
cfg['_group'] = 'with_pattern_rec'
groupcaption = 'date:20231016'
# 一个group是一系列的调参，应当保证一个group的数据集和测试性能的代码基本一致
#_record/{_group}-|cfgstrs.txt
#                 |code/
#                 |xxres/
#                 |xxres/eval.txt
#                 |xxres/ckpt/
groupdir = f'/home/wtxt/a/_record/{cfg["_group"]}'
xxres = create_path(groupdir,f'{exec_id}_res')
cfgstrs,code = create_path(groupdir,'cfgstrs.txt'),create_path(groupdir,'code')
evalpath = create_path(xxres,'eval.txt')
ckptdir = create_path(xxres,'ckpt')
imgdir = create_path(xxres,'imgs')
tiffdir = create_path(xxres,'tiffs')
eval = Eval(evalpath)
print(xxres,cfgstrs,code,evalpath,ckptdir)
if os.path.exists(cfgstrs):
    with open(cfgstrs,'r') as f:
        basecfg = f.readlines()[-1] #只要保证最后一行写的是cfg就可以了，中间行可以随意添加
        assert '_trainsize' in basecfg #防止由于上一次意外退出导致最后一行不是cfg
else:
    with open(cfgstrs,'w',encoding='utf8') as f:
        f.write(groupcaption+'\n')
    basecfg = 'preiod-10_lr-0.0003_sigma-0.5_trainsize-99_validsize-5_bs-3_features-32_optim-adam_cropsize-256' #memory 32.6G

cfg = update(cfg,basecfg)
cfg = update(cfg,sys.argv[-1])
with open(cfgstrs,'a') as f:
    f.write('\n'+dict2str(cfg))
# scalers = ['trainedsize','time','valloss','rec_loss','nrmse','pattern_loss','trainloss']
# evaltxt里面的可选项，这些是随时间T的变量，这里有time和trainedsize变量是因为test的时间点不固定（考虑到时间成本）,就不要考虑文件读写的时间成本了。
print(cfg)
# coco_bg = jnp.load('/data_nas/nas/Research/dlold/data/cocosim.npy')
def iterwrapfn(train_files,cropsize,bs):
    arr = tiff.imread(train_files) #可以读文件列表或者单个文件
    np.random.shuffle(arr)
    arr = arr[:arr.shape[0]//bs*bs]
    ### random crop
    res = []
    cropcood = np.random.randint(0,256,size=(arr.shape[0],2),)
    # cropcood = jax.random.randint(rng,(arr.shape[0],2),0,256) #会卡，不知什么原因
    for idx in range(arr.shape[0]):
        i,j = cropcood[idx]
        res.append(arr[idx,i:i+cropsize,j:j+cropsize])
    res = jnp.stack(res).reshape(-1,bs,1,cropsize,cropsize)
    print(time.time()-start_time)
    return res

from network import DND_SIM
net = DND_SIM(cfg['features'])
optimizer_name = cfg['optim']
num_steps = 2e7 // cfg['bs']
optimizer_hparams = {'lr':cfg['lr']}
class TrainState(train_state.TrainState):
    rng : Any = None
def create_train_state(net):
    rng = jax.random.PRNGKey(0)
    x_init = jnp.ones((cfg['bs'],9,cfg['cropsize'],cfg['cropsize']))
    variables = net.init({'params':rng,'dropout':rng}, x_init)
    if optimizer_name.lower() == 'adam':
        opt_class = optax.adam
    elif optimizer_name.lower() == 'adamw':
        opt_class = optax.adamw
    elif optimizer_name.lower() == 'sgd':
        opt_class = optax.sgd
    else:
        assert False, f'Unknown optimizer "{opt_class}"'
    # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=optimizer_hparams.pop('lr'),
        boundaries_and_scales=
            {int(num_steps*0.6): 0.1,
                int(num_steps*0.85): 0.1}
    )
    # Clip gradients at max value, and evt. apply weight decay
    transf = [optax.clip(1.0)]
    # transf = [optax.clip(1.0)]
    if opt_class == optax.sgd and 'weight_decay' in optimizer_hparams:  # wd is integrated in adamw
        transf.append(optax.add_decayed_weights(optimizer_hparams.pop('weight_decay')))
    optimizer = optax.chain(
        *transf,
        opt_class(lr_schedule, **optimizer_hparams)
        )
    return TrainState.create(
        apply_fn=net.apply, params=variables['params'], tx=optimizer, rng=rng)
@jax.jit
def train_step(state, batch):
    rng,_ = jax.random.split(state.rng,2)
    loss_fn = lambda params: compute_metrics(batch, net, params, rng, train=True)
    # Get loss, gradients for loss, and other outputs of loss function
    ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    loss, (error,res) = ret[0],ret[1]
    # Update parameters and batch statistics
    state = state.apply_gradients(grads=grads,rng=rng)
    return error, res, state
state = create_train_state(net)
@jax.jit
def eval_step(state,batch):
    loss, (error, res) = compute_metrics(batch, net, state.params, state.rng, False)
    return error,res
from utils.utils_phys import standard_psf_debye, pattern_generation_jax
def noise_generation(cfg,key_new,):
    float_noise = jax.random.uniform(key_new,shape=(cfg['bs'],1,1,1), minval=0, maxval=1)
    noise = jax.random.normal(key_new, (cfg['bs'],9,cfg['cropsize'],cfg['cropsize'])) * float_noise *cfg['sigma']
    return noise
train_files = glob.glob(f'/home/wtxt/a/data/vo_circle_sim/30/train_gt/*.tif')
test_files = glob.glob(f'/home/wtxt/a/data/vo_circle_sim/30/test_gt/*.tif')
# train_files = '/dataf/ndl/_record/newdataset/1000-1000-100-20-merge_id-onj.tif'
# test_files = '/dataf/ndl/_record/newdataset/100-1000-100-20-merge_id-ond.tif'
dataset_ts = iterwrapfn(test_files,cfg['cropsize'],cfg['bs'])


trained = 0
from tqdm import tqdm
for epoch in range(10000):
    if epoch % 50 == 0:#  and epoch>10
        print("start evaluation")
        eval_start_time = time.time()
        traindir = xxres
        checkpoints.save_checkpoint(ckpt_dir=ckptdir,
                            target={'params': state.params,
                                    'rng': state.rng},
                            step=trained,
                            overwrite=True)        
        metrics_eval = {'rec': [], 'rec_p': [], 'nrmse': [], 'loss':[]}
        for x in dataset_ts:
            noise,I = noise_generation(cfg,jax.random.PRNGKey(0)),pattern_generation_jax(cfg,jax.random.PRNGKey(0))
            I = twist(I)
            batch = (x,I,noise)
            metrics,res = eval_step(state,batch)
            metrics_eval = {k: v + [metrics[k]] for k, v in metrics_eval.items()}
        metrics_eval = {k: sum(v)/len(v) for k, v in metrics_eval.items()}
        print('\n',"%.2e"%np.array(metrics_eval['rec'].mean()),"%.2e"%np.array(metrics_eval['nrmse'].mean()))
        # test(state)
        eval.trained,eval.time,eval.loss,eval.rec,eval.nrmse,eval.rec_p = \
            trained,int(time.time()-start_time),metrics_eval['loss'].item()*100,metrics_eval['rec'].item()*100,metrics_eval['nrmse'].item()*100,metrics_eval['rec_p'].item()*100
        eval.save()
        # with open(f'{traindir}/scaler.txt','a') as f:
        #     f.write('\t'.join([str(x) for x in [trained,int(time.time()-start_time),metrics_eval['loss'],trainloss]])+'\n') 
            # ['trainedsize','time','valloss','rec_loss','nrmse','pattern_loss','trainloss']
        savetif(res['rec'],create_path(tiffdir,printinfo(ext='.tif')))
        savetif(noise,create_path(tiffdir,printinfo(ext='.tif')))
        savetif(x,create_path(tiffdir,printinfo(ext='.tif')))
        savetif(I,create_path(tiffdir,printinfo(ext='.tif')))
        savetif(standard_psf_debye(31),create_path(tiffdir,printinfo(ext='.tif')))
        savetif(res['rec_p'],create_path(tiffdir,printinfo(ext='.tif')))
        savetif(res['D'],create_path(tiffdir,printinfo(ext='.tif')))
        for i in range(2):
            savepng(x[i,0],create_path(imgdir,printinfo(ext='.png')))
            savepng(res['D'][i].mean(axis=0),create_path(imgdir,printinfo(ext='.png')))
            savepng(res['rec'][i,0],create_path(imgdir,printinfo(ext='.png')))
            savepng(I[i,:].reshape(3,3,256,256).transpose(0,2,1,3).reshape(3*256,3*256),create_path(imgdir,printinfo(ext='.png')))
            savepng(res['rec_p'][i,:].reshape(3,3,256,256).transpose(0,2,1,3).reshape(3*256,3*256),create_path(imgdir,printinfo(ext='.png')))
            time.sleep(1)
        print('eval consume seconds: ',int(time.time()-eval_start_time))
    rng = state.rng
    dataset = iterwrapfn(train_files[:cfg['trainsize']],cfg['cropsize'],cfg['bs'])
    for i,x in tqdm(enumerate(dataset)):
        # print(trained,int(time.time()-start_time))
        trained += x.shape[0]
        noise,I = noise_generation(cfg,state.rng),pattern_generation_jax(cfg,state.rng,),
        I = twist(I)
        batch = (x,I,noise)
        error, res, state = train_step(state,batch)
        trainloss = error['loss'] if i==0 else trainloss*i/(i+1) + error['loss']/(i+1)
    eval.trainloss = trainloss.item()
os._exit(0)





