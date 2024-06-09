import os
import importlib
from PIL import Image
from tqdm import tqdm
from utils.test_option import args
import numpy as np
import torch
from data import create_loader
from model.unet import UNet
from loss.loss import logits_to_binary_segmentation


def main_worker(args):
    # setup data set and data loader
    dataloader, dataloader_generator = create_loader(args)

    # Model and version
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()

    segnet = UNet(3, 1, bilinear=False).cuda()
    segnet.load_state_dict(torch.load(args.pre_train_seg, map_location='cuda'))
    segnet.eval()

    os.makedirs(args.outputs, exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'comp_results'), exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'gts'), exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'segs'), exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'full_segs'), exist_ok=True)

    filename_dict = dict()
    for idx in tqdm(range(args.num_test // args.batch_size)):
        image, mask, filename = next(dataloader_generator)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask).float() + mask
        with torch.no_grad():
            pred_img = model(image_masked, mask)

        comp_imgs = (1 - mask) * image + mask * pred_img
        full_seg = segnet(comp_imgs[:, :3, :, :])
        for i in range(image.shape[0]):
            mask_tmp = (255 * mask[i][0].detach().cpu().numpy()).astype(np.uint8)
            seg_tmp = (255 * image[i, 3, ...].detach().cpu().numpy()).astype(np.uint8)
            full_seg_tmp = (255 * logits_to_binary_segmentation(full_seg[i][0]).detach().cpu().numpy()).astype(np.uint8)
            comp_imgs_tmp = (
                        255 * ((comp_imgs[i, :3, ...] + 1.0) / 2.0).permute(1, 2, 0).detach().cpu().numpy()).astype(
                np.uint8)
            image_tmp = (255 * ((image[i, :3, ...] + 1.0) / 2.0).permute(1, 2, 0).detach().cpu().numpy()).astype(
                np.uint8)
            filename_tmp = filename[i].split('.')[0]
            if filename_tmp in filename_dict.keys():
                filename_dict[filename_tmp] += 1
            else:
                filename_dict[filename_tmp] = 1
            # Save Images
            Image.fromarray(mask_tmp).save(
                os.path.join(args.outputs, 'masks', f'{filename_tmp}_{filename_dict[filename_tmp]}_mask.png'))
            Image.fromarray(comp_imgs_tmp).save(
                os.path.join(args.outputs, 'comp_results', f'{filename_tmp}_{filename_dict[filename_tmp]}.jpg'))
            Image.fromarray(image_tmp).save(
                os.path.join(args.outputs, 'gts', f'{filename_tmp}_{filename_dict[filename_tmp]}.jpg'))
            Image.fromarray(seg_tmp).save(
                os.path.join(args.outputs, 'segs', f'{filename_tmp}_{filename_dict[filename_tmp]}.jpg'))
            Image.fromarray(full_seg_tmp).save(
                os.path.join(args.outputs, 'full_segs', f'{filename_tmp}_{filename_dict[filename_tmp]}.jpg'))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main_worker(args)
