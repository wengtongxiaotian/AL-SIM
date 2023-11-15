import glob
import numpy as np
import jax
import jax.numpy as jnp
from skimage.io import imread
from skimage.transform import resize
from torchvision import transforms
import torch
from torch import Tensor
from torchvision.transforms.transforms import _setup_size, F
from typing import Tuple
import random
class dataset_lf(torch.utils.data.Dataset):
    def __init__(self, file_paths, crop_size, sample_size=None):
        super().__init__()
        self.file_paths = file_paths
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.picnum = len(file_paths)
        self.crop_size = crop_size
        
    def __getitem__(self, idx):
        lfs = random.sample(self.file_paths, 3)
        I = []
        for i in range(3):
            coco_item = lfs[i]
            coco_item = imread(coco_item, as_gray=True)
            coco_item -= coco_item.min()
            coco_item = coco_item/coco_item.max()
            phi = coco_item*np.pi*2
            I1 = []
            phi0 = np.random.rand()*np.pi
            for j in range(3):
                I1.append(np.sin(phi+j*np.pi/3*2+phi0)+1)
            I1 = np.stack(I1)
            I1 = resize(I1,(3,self.crop_size,self.crop_size))
            I1 = I1/I1.mean(axis=0,keepdims=True) # consider rec_p is from softmax,mean won't make sense
            I.append(I1)
        x = np.concatenate(I)
        # x = self.transform(x)
        return x

    def __len__(self):
        return 10000000

class dataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, crop_size, sample_size=None):
        super().__init__()
        self.file_paths = file_paths
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                MultiChannelRandomCrop(size=crop_size, pad_if_needed=True)
            ]
        )
        self.sample_size = sample_size
        self.indices = np.random.permutation(len(self.file_paths))
        
    def __getitem__(self, idx):
        x = imread(self.file_paths[self.indices[idx]], as_gray=True).astype(np.float32)
        x = self.transform(x)
        mean = x.mean(axis=(-2, -1), keepdims=True)
        var = x.var(axis=(-2, -1), keepdims=True)
        x = (x - mean) / (var + 1e-6) ** .5
        return x

    def __len__(self):
        if self.sample_size is not None:
            return self.sample_size
        else:
            return len(self.file_paths)


class psfset(torch.utils.data.Dataset):
    def __init__(self, psf_size, psf_num):
        super().__init__()
        self.psf_size = psf_size
        self.psf_num = psf_num

    def __getitem__(self, idx):
        rng = jax.random.PRNGKey(idx)
        psf = psf_generator(self.psf_size, rng)
        return np.array(psf)

    def __len__(self):
        return self.psf_num



def get_sample(dataloader):
    for x in dataloader:
        return jnp.array(x)




class MultiChannelRandomCrop(torch.nn.Module):
    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (Tensor): Image to be cropped CYX.
            output_size (tuple): Expected output size of the crop [y, x].
        Returns:
            tuple: params (c, y, x) to be passed to ``crop`` for random crop.
        """
        w, h = F.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
    
    def pad_img(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        return img

    def forward(self, imgs):
        """
        Args:
            imgs (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        imgs = [self.pad_img(img) for img in imgs]
        i, j, h, w = self.get_params(imgs[0], self.size)
        imgs = [F.crop(img, i, j, h, w) for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        return imgs