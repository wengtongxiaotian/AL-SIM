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