from skimage.io import imread,imsave
path = '/dataf/Research/Jax-AI-for-science/SSL-SIM/ckpt/clip_finetune/sim_data/crop_size=[1, 224, 224]--eval_sigma=None--add_noise=1--decay_steps_ratio=0.9--mask_ratio=0.75--patch_size=[1, 16, 16]--psf_size=[1, 49, 49]--rescale=[1, 1, 1]--stage_1/psf.tif'
import pdb;pdb = pdb.set_trace
img = imread(path)
pdb()
imsave('tmp.jpg',img)
print()


