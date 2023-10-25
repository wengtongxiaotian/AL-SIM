import sys
sys.path.append('.')
if len(sys.argv) == 1:
    sys.argv.append(f'group-coco_cuda-1')#_onlypattern
from tool import np,update,dict2str,create_path,tiff,glob,savetif,savepng,printinfo,exec_dir,saveresult,global_dict,tikcount
from data_simulation.distort import twist
from train_utils import simple_score as compute_metrics
from ai_codes.eval_class import Eval
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
from typing import Any
# 一个group是一系列的调参，应当保证一个group的数据集和测试性能的代码基本一致
#_record/{_group}-|exec_dir/ # this is exec_dir
#                 |exec_dir/eval.txt
#                 |exec_dir/ckpt/
evalpath = create_path(exec_dir,'eval.txt')
ckptdir = create_path(exec_dir,'ckpt')
imgdir = create_path(exec_dir,'imgs')
tiffdir = create_path(exec_dir,'tiffs')
print(exec_dir,evalpath,ckptdir)
basecfg = 'preiod-10_lr-0.0003_sigma-0.5_trainsize-99_validsize-5_bs-3_features-32_optim-adam_cropsize-256' #memory 32.6G
basecfg = 'preiod-10_lr-0.0003_sigma-0.5_trainsize-6000_validsize-90_bs-10_features-128_optim-adam_cropsize-256_mode-vscode_caption-sigma0.03'
cfg = update({},basecfg)
cfg = update(cfg,sys.argv[-1])
eval = Eval(evalpath,dict2str(cfg))
# scalers = ['trainedsize','time','valloss','rec_loss','nrmse','pattern_loss','trainloss']
# evaltxt里面的可选项，这些是随时间T的变量，这里有time和trainedsize变量是因为test的时间点不固定（考虑到时间成本）,就不要考虑文件读写的时间成本了。
print(cfg)
# coco_bg = jnp.load('/data_nas/nas/Research/dlold/data/cocosim.npy')
def norm(arr): #(bs,h,w)
    mean = arr.mean(axis=(-4,-3,-2,-1),keepdims=True)
    var = arr.var(axis=(-4,-3,-2,-1),keepdims=True)
    return (arr-mean)/(var+1e-6)**.5
def iterwrapfn(train_files,cropsize,bs):
    arr = tiff.imread(train_files) #可以读文件列表或者单个文件
    np.random.shuffle(arr)
    arr = arr[:arr.shape[0]//bs*bs]
    imgsize = arr.shape[-1]
    ### random crop
    res = []
    cropcood = np.random.randint(0,imgsize-256,size=(arr.shape[0],2),)
    # cropcood = jax.random.randint(rng,(arr.shape[0],2),0,256) #会卡，不知什么原因
    for idx in range(arr.shape[0]):
        i,j = cropcood[idx]
        res.append(arr[idx,i:i+cropsize,j:j+cropsize])
    res = jnp.stack(res).reshape(-1,bs,1,cropsize,cropsize)
    res = norm(res)
    print(printinfo())
    return res
I_data = tiff.imread('/dataf/b/_record/awareligthfield/qz/test55-2-pattern.tif')
def xx(key_new):
    I = np.zeros((cfg['bs'],9,cfg['cropsize'],cfg['cropsize']))
    for i in range(cfg['bs']):
        ip,jp = np.random.randint(256,2048-256,size=(2,),)
        I[i] = I_data[:,ip:ip+256,jp:jp+256]
    M = jax.random.uniform(key_new, shape=(cfg['bs'],3,1,1), minval=0.2, maxval=0.8)
    I = I*jnp.repeat(M,3,axis=1)
    I = I / jnp.mean(I, axis=(-2, -1), keepdims=True)
    return I
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
train_files = '/dataf/ndl/_record/newdataset/coco_bg.tif'
test_files = '/dataf/ndl/_record/newdataset/coco_bg.tif'
dataset_ts = iterwrapfn(test_files,cfg['cropsize'],cfg['bs'])[-cfg['validsize']:]


trained = 0
from tqdm import tqdm
for epoch in range(10000):
    if epoch % 1 == 0:#  and epoch>10
        print("start evaluation",printinfo())
        traindir = exec_dir
        checkpoints.save_checkpoint(ckpt_dir=ckptdir,
                            target={'params': state.params,
                                    'rng': state.rng},
                            step=trained,
                            overwrite=True)        
        metrics_eval = {'rec': [], 'rec_p': [], 'nrmse': [], 'loss':[]}
        for x in dataset_ts:
            noise = noise_generation(cfg,jax.random.PRNGKey(0))
            if not global_dict.get('realdata',False):
                I = pattern_generation_jax(cfg,jax.random.PRNGKey(0))
                I = twist(I)
            else: I = xx(jax.random.PRNGKey(0))
            batch = (x,I,noise)
            metrics,res = eval_step(state,batch)
            metrics_eval = {k: v + [metrics[k]] for k, v in metrics_eval.items()}
        metrics_eval = {k: sum(v)/len(v) for k, v in metrics_eval.items()}
        print('\n',"%.2e"%np.array(metrics_eval['rec'].mean()),"%.2e"%np.array(metrics_eval['nrmse'].mean()))
        # region save
        eval.trained,eval.time,eval.loss,eval.rec,eval.nrmse,eval.rec_p = \
            trained,tikcount(),metrics_eval['loss'].item()*100,metrics_eval['rec'].item()*100,metrics_eval['nrmse'].item()*100,metrics_eval['rec_p'].item()*100
        eval.save()
        # saveresult(f"pattern_loss: {round(metrics_eval['rec_p'].item()*100,3)}")
        saveresult(f"rec_loss: {round(metrics_eval['rec'].item()*100,3)}")
        savetif(res['rec'],create_path(tiffdir,printinfo(1)+'.tif'))
        savetif(noise,create_path(tiffdir,printinfo(1)+'.tif'))
        savetif(x,create_path(tiffdir,printinfo(1)+'.tif'))
        savetif(I,create_path(tiffdir,printinfo(1)+'.tif'))
        savetif(standard_psf_debye(31),create_path(tiffdir,printinfo(1)+'.tif'))
        savetif(res['rec_p'],create_path(tiffdir,printinfo(1)+'.tif'))
        savetif(res['D'],create_path(tiffdir,printinfo(1)+'.tif'))
        for i in range(2):
            savepng(x[i,0],create_path(imgdir,printinfo(1)+'.png'))
            savepng(res['D'][i].mean(axis=0),create_path(imgdir,printinfo(1)+'.png'))
            savepng(res['rec'][i,0],create_path(imgdir,printinfo(1)+'.png'))
            savepng(I[i,:].reshape(3,3,256,256).transpose(0,2,1,3).reshape(3*256,3*256),create_path(imgdir,printinfo(1)+'.png'))
            savepng(res['rec_p'][i,:].reshape(3,3,256,256).transpose(0,2,1,3).reshape(3*256,3*256),create_path(imgdir,printinfo(1)+'.png'))
        print("end evaluation",printinfo())
        # endregion
    
    rng = state.rng
    dataset = iterwrapfn(train_files[:cfg['trainsize']],cfg['cropsize'],cfg['bs'])
    for i,x in tqdm(enumerate(dataset)):
        # print(trained,int(time.time()-start_time))
        trained += x.shape[0]
        noise = noise_generation(cfg,state.rng)
        if not global_dict.get('realdata',False):
            I = pattern_generation_jax(cfg,state.rng,),
            I = twist(I)
        else:I = xx(state.rng)
        batch = (x,I,noise)
        error, res, state = train_step(state,batch)
        trainloss = error['loss'] if i==0 else trainloss*i/(i+1) + error['loss']/(i+1)
    eval.trainloss = trainloss.item()





