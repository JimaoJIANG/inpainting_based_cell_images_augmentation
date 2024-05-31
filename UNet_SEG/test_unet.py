import torch
import torch.optim
from dataloader import MoNuSegTestDataset
from unet import UNet
from torch.utils.data import Dataset
from torch import utils
import numpy as np
from PIL import Image
import os

def dice_coefficient(segmap1, segmap2):
    segmap1 = segmap1.bool()
    segmap2 = segmap2.bool()
    intersection = torch.logical_and(segmap1, segmap2)
    return 2. * intersection.sum().float() / (segmap1.sum().float() + segmap2.sum().float())

def logits_to_binary_segmentation(logits):
    probs = torch.sigmoid(logits)
    binary_segmentation = (probs > 0.5).float()
    return binary_segmentation


def Test_Unet(net, device, data_path_test, save_path, batch_size=1):
    test_dataset = MoNuSegTestDataset(data_path_test, with_name=True)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    net.eval()
    dices = list()
    for idx, (gt, label, name) in enumerate(test_loader):
        gt = gt.to(device=device, dtype=torch.float32)
        # gt = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(gt)
        gt = gt + 0.1 * torch.rand(gt.shape).to(device)
        label = label.to(device=device, dtype=torch.float32)
        pred_real = net(gt)
        binary_pred = logits_to_binary_segmentation(pred_real)
        dice = dice_coefficient(binary_pred, label)
        dices.append(dice.item())
        Image.fromarray(255 * (binary_pred[0][0].detach().cpu().numpy()).astype(np.uint8)).save(
            os.path.join(save_path, name[0] + '.png'))
    print('dice:', torch.mean(torch.tensor(dices)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Image Inpainting')
    parser.add_argument('--data_path_test', type=str, required=True,
                        help='images for testing, directing to imgs directory of the testing images')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='path to pretrained models')
    parser.add_argument('--save_path', type=str, default='./outputs',
                        help='path to save results')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size in each mini-batch')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(3, 1, bilinear=False)
    net.to(device=device)
    net.load_state_dict(torch.load(args.pretrained, map_location=device))
    data_path_test = "/home/pc/diskB/jjm/cell/dataset/test/imgs/"
    Test_Unet(net, device, data_path_test, args.batch_size)
