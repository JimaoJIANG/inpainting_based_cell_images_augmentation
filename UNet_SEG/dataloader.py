import random
import numpy as np
import torch.utils.data.dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image, ImageSequence
import torchvision
import os

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif')

def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)

class MoNuSegDataset(torch.utils.data.dataset.Dataset):
    """
    A dataset class that samples 2D slices from 3D volumes and applies data augmentation.

    Args:
        cfg (dict): A dictionary containing configuration parameters for the dataset.

    Attributes:
        path_list (list): A list of paths to the input data files.
        pad (int): The amount of padding to add to the input data.
        aug (str): A string indicates data augmentation to the input data.
        img_size (int): The size of the 2D slices to sample from the 3D volumes.
        key_name_dict (dict): A dictionary mapping file paths to the names of the tensors in the data files.
        data_list (list): A list of tensors representing the input data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a randomly sampled 2D slice from the input data at the given index.
        sample_cord(data_idx, axis): Samples a 2D slice from the input data at the given index and axis.

    """

    def __init__(self, path, img_size=256, mask_dir='masks', comp_dir='comp_results', gt_dir='gts', label_dir='segs'):
        self.img_size = img_size
        self.key_name_dict = {}
        self.comps_list = []
        self.masks_list = []
        self.gts_list = []
        self.gts_labels_list = []
        tt = ToTensor()

        for root, _, fnames in os.walk(path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    path_name= os.path.join(root, fname)
                    comp_array = np.array(Image.open(path_name))
                    H, W, _ = comp_array.shape
                    comp_tensor = tt(comp_array)
                    self.comps_list.append(comp_tensor)

                    mask_name = path_name.replace(comp_dir, mask_dir).replace('.jpg', '_mask.png')
                    mask_img_array = np.array(Image.open(mask_name))
                    mask_img_tensor = tt(mask_img_array)
                    self.masks_list.append(mask_img_tensor)

                    gt_name = path_name.replace(comp_dir, gt_dir)
                    gt_img_array = np.array(Image.open(gt_name))
                    gt_img_tensor = tt(gt_img_array)
                    self.gts_list.append(gt_img_tensor)

                    label_name = path_name.replace(comp_dir, label_dir)
                    label_img_array = np.array(Image.open(label_name))
                    label_img_tensor = tt(label_img_array)
                    self.gts_labels_list.append(label_img_tensor)


    def __len__(self):
        """
        Returns the number of samples in each epoch.
        """
        return len(self.comps_list)

    def __getitem__(self, idx):
        """
        Returns a randomly sampled 2D slice from the input data at the given index.
        Returns:
            A tensor representing a randomly sampled 2D slice from the input data.
        """
        # curr_data_idx = random.randrange(0, len(self.data_list)) # select dataset
        # return self.sample_cord(curr_data_idx) # return 2D slice
        return self.sample_cord(idx)

    def _transform(self, image, mask, gt, label):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[1])
            mask = torch.flip(mask, dims=[1])
            gt = torch.flip(gt, dims=[1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
            mask = torch.flip(mask, dims=[2])
            gt = torch.flip(gt, dims=[2])
        angle = random.choice([0, 90, 180, 270])
        image = torchvision.transforms.functional.rotate(image, angle)
        label = torchvision.transforms.functional.rotate(label, angle)
        mask = torchvision.transforms.functional.rotate(mask, angle)
        gt = torchvision.transforms.functional.rotate(gt, angle)
        return image, mask, gt, label

    def sample_cord(self, data_idx):
        """
        Samples a 2D slice from the input data at the given index and axis.

        Args:
            data_idx (int): The index of the input data to sample from.

        Returns:
            A tensor representing a 2D slice sampled from the input data at the given index and axis.
        """
        comp = self.comps_list[data_idx] # get dataset
        label = self.gts_labels_list[data_idx] # get dataset
        gt = self.gts_list[data_idx] # get dataset
        mask = self.masks_list[data_idx] # get dataset

        _, d_x, d_y = comp.shape
        if d_x > self.img_size + 1:
            x_sample = torch.randint(
                low=0, high=int(d_x - self.img_size - 1), size=(1,)
            )
        else:
            x_sample = 0
        if d_y > self.img_size + 1:
            y_sample = torch.randint(
                low=0, high=int(d_y - self.img_size - 1), size=(1,)
            )
        else:
            y_sample = 0
        comp_sample = comp[:, x_sample: x_sample + self.img_size, y_sample: y_sample + self.img_size]
        label_sample = label[:, x_sample: x_sample + self.img_size, y_sample: y_sample + self.img_size]
        mask_sample = mask[:, x_sample: x_sample + self.img_size, y_sample: y_sample + self.img_size]
        gt_sample = gt[:, x_sample: x_sample + self.img_size, y_sample: y_sample + self.img_size]
        return self._transform(comp_sample, mask_sample, gt_sample, label_sample)


class MoNuSegTestDataset(torch.utils.data.dataset.Dataset):
    """
    A dataset class that samples 2D slices from 3D volumes and applies data augmentation.

    Args:
        cfg (dict): A dictionary containing configuration parameters for the dataset.

    Attributes:
        path_list (list): A list of paths to the input data files.
        pad (int): The amount of padding to add to the input data.
        aug (str): A string indicates data augmentation to the input data.
        img_size (int): The size of the 2D slices to sample from the 3D volumes.
        key_name_dict (dict): A dictionary mapping file paths to the names of the tensors in the data files.
        data_list (list): A list of tensors representing the input data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns a randomly sampled 2D slice from the input data at the given index.
        sample_cord(data_idx, axis): Samples a 2D slice from the input data at the given index and axis.

    """

    def __init__(self, path, img_size=1000, gt_dir='imgs', label_dir='segs', with_name=False):
        self.img_size = img_size
        self.key_name_dict = {}
        self.gts_list = []
        self.gts_labels_list = []
        self.file_names = []
        self.with_name = with_name
        tt = ToTensor()

        for root, _, fnames in os.walk(path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    path_name = os.path.join(root, fname)
                    gt_name = path_name
                    gt_img_array = np.array(Image.open(gt_name))
                    gt_img_tensor = tt(gt_img_array)
                    self.gts_list.append(gt_img_tensor)

                    label_name = path_name.replace(gt_dir, label_dir).replace('.tif', '.png')
                    label_img_array = np.array(Image.open(label_name))
                    label_img_tensor = tt(label_img_array)
                    self.gts_labels_list.append(label_img_tensor)
                    self.file_names.append(path_name.split('/')[-1].split('.')[0])

    def __len__(self):
        """
        Returns the number of samples in each epoch.
        """
        return len(self.gts_list)

    def __getitem__(self, idx):
        """
        Returns a randomly sampled 2D slice from the input data at the given index.
        Returns:
            A tensor representing a randomly sampled 2D slice from the input data.
        """
        # curr_data_idx = random.randrange(0, len(self.data_list)) # select dataset
        # return self.sample_cord(curr_data_idx) # return 2D slice
        return self.sample_cord(idx)
    def sample_cord(self, data_idx):
        """
        Samples a 2D slice from the input data at the given index and axis.

        Args:
            data_idx (int): The index of the input data to sample from.

        Returns:
            A tensor representing a 2D slice sampled from the input data at the given index and axis.
        """
        label = self.gts_labels_list[data_idx] # get dataset
        gt = self.gts_list[data_idx] # get dataset

        _, d_x, d_y = gt.shape
        if d_x > self.img_size + 1:
            x_sample = torch.randint(
                low=0, high=int(d_x - self.img_size - 1), size=(1,)
            )
        else:
            x_sample = 0
        if d_y > self.img_size + 1:
            y_sample = torch.randint(
                low=0, high=int(d_y - self.img_size - 1), size=(1,)
            )
        else:
            y_sample = 0
        label_sample = label[:, x_sample: x_sample + self.img_size, y_sample: y_sample + self.img_size]
        gt_sample = gt[:, x_sample: x_sample + self.img_size, y_sample: y_sample + self.img_size]
        if self.with_name:
            return gt_sample, label_sample, self.file_names[data_idx]
        return gt_sample, label_sample