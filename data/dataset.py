import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from data.mask_generator import RandomMask

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif')

def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)


class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type

        # image and mask
        self.image_path = []
        self.seg_path = []
        self.paths = [entry.path for entry in os.scandir(args.dir_image)
                          if is_image_file(entry.name)]
        self.seg_paths = [entry.path.replace('imgs', 'segs').replace('tif', 'png') for entry in os.scandir(args.dir_image)
                      if is_image_file(entry.name)]
        self.image_path.extend(self.paths)
        self.seg_path.extend(self.seg_paths)

        # augmentation
        if args.transform == 'randomcrop':
            self.img_trans_only = transforms.Compose([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                           std=(0.5, 0.5, 0.5))])
            self.img_trans = transforms.Compose(
                [transforms.RandomResizedCrop(args.crop_size, antialias=True),
                 transforms.RandomHorizontalFlip()
                 ]
            )
        elif args.transform == 'centercrop':
            self.img_trans_only = transforms.Compose([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                           std=(0.5, 0.5, 0.5))])
            self.img_trans = transforms.Compose(
                [transforms.CenterCrop(args.crop_size),
                 transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                 transforms.RandomHorizontalFlip()
                 ]
            )
        elif args.transform == 'resize_and_crop':
            self.img_trans_only = transforms.Compose([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                           std=(0.5, 0.5, 0.5))])
            self.img_trans = transforms.Compose(
                [transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                 transforms.CenterCrop(args.crop_size),
                 transforms.RandomHorizontalFlip(),
                 ]
            )
        else:
            raise NotImplementedError("Image transformation type %s is not implemented!" % args.transform)

        self.mask_trans = transforms.Compose([
            transforms.Resize(args.crop_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        self.mask_trans_simple = transforms.Compose([
            transforms.Resize(args.crop_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        seg = Image.open(self.seg_path[index]).convert('L')
        seg = transforms.ToTensor()(seg)
        image = self.img_trans_only(image)
        image_seg = torch.cat([image, seg], dim=0)
        image_seg = self.img_trans(image_seg)
        image_seg[3, :, :] = (image_seg[3, :, :] > 0.5).to(torch.float32)
        filename = os.path.basename(self.image_path[index])

        if self.mask_type == 'centered':
            mask = np.zeros((self.h, self.w)).astype(np.uint8)
            mask[self.h // 4:self.h // 4 * 3, self.w // 4:self.w // 4 * 3] = 1
            mask = Image.fromarray(mask).convert('L')
            mask = self.mask_trans(mask)
        elif self.mask_type == 'random':
            mask = Image.fromarray(RandomMask(self.h, hole_range=[0, 0.8])).convert('L')
            mask = self.mask_trans_simple(mask)
        return image_seg, mask, filename


class MoNuSegTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path, img_size=1000, gt_dir='imgs', label_dir='segs'):
        self.img_size = img_size
        self.key_name_dict = {}
        self.gts_list = []
        self.gts_labels_list = []
        tt = transforms.ToTensor()

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


    def __len__(self):
        """
        Returns the number of samples in each epoch.
        """
        return len(self.gts_list)

    def __getitem__(self, idx):
        return self.sample_cord(idx)
    def sample_cord(self, data_idx):
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
        gt_sample = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(gt_sample)
        return gt_sample, label_sample

def get_test_loader(args):
    dataset = MoNuSegTestDataset(args.test_path, args.test_img_size)
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)