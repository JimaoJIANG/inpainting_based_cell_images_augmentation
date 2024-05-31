import torch
import torch.optim
from dataloader import MoNuSegDataset
from log import Logger
import torch.nn as nn
from unet import UNet
from torch.utils.data import Dataset
from torch import utils
import time

Unet_train_txt = Logger('Unet_train.txt')

def dice_coefficient(segmap1, segmap2):
    segmap1 = segmap1.bool()
    segmap2 = segmap2.bool()
    intersection = torch.logical_and(segmap1, segmap2)
    return 2. * intersection.sum().float() / (segmap1.sum().float() + segmap2.sum().float())

def logits_to_binary_segmentation(logits):
    probs = torch.sigmoid(logits)
    binary_segmentation = (probs > 0.5).float()
    return binary_segmentation

def loss_fun_gen(pred, gt, mask, device):
    pred_unmasked = pred * (1 - mask)
    gt_unmasked = gt * (1 - mask)
    loss_gt = nn.functional.binary_cross_entropy_with_logits(pred_unmasked, gt_unmasked)
    pred_masked = pred * mask
    ones_mask = torch.ones_like(pred_masked).to(device)
    zeros_mask = torch.zeros_like(pred_masked).to(device)
    loss_reg = nn.functional.binary_cross_entropy_with_logits(pred_masked, ones_mask) +\
               nn.functional.binary_cross_entropy_with_logits(pred_masked, zeros_mask)
    loss = loss_gt + 0.1 * loss_reg
    return loss

def Train_Unet(net, device, data_path_train, batch_size=16, epochs=200, lr=0.0001, test_epoch=1):
    train_dataset = MoNuSegDataset(data_path_train)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95))
    loss_fun_real = nn.BCEWithLogitsLoss()
    best_los = float('inf')
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        i = 0
        begin = time.perf_counter()
        for comp, mask, gt, label in train_loader:
            opt.zero_grad()
            comp = comp.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)
            gt = gt.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # Real samples
            pred_real = net(gt)
            loss_real = loss_fun_real(pred_real, label)
            # Generated samples
            # comp_white = gt * (1 - mask) + mask * torch.ones_like(gt).to(device)
            # comp_random = gt * (1 - mask) + mask * torch.rand(gt.shape).to(device)
            # comp_black = gt * (1 - mask)
            pred_gen = net(comp)
            loss_gen = loss_fun_gen(pred_gen, label, mask, device)

            # Overall loss
            loss = loss_real + 1.0 * loss_gen
            loss.backward()
            i = i + 1
            running_loss = running_loss + loss.item()
            opt.step()
        end = time.perf_counter()
        loss_avg_epoch = running_loss / i
        Unet_train_txt.write(str(format(loss_avg_epoch, '.4f')) + '\n')
        print('epoch: %d avg loss: %f time:%d s' % (epoch, loss_avg_epoch, end - begin))
        if loss_avg_epoch < best_los:
            best_los = loss_avg_epoch
            torch.save(net.state_dict(), 'model_pth.pkl')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Image Inpainting')
    parser.add_argument('--data_path_train', type=str, required=True,
                        help='images for training, directing to comp_results directory of the generated path')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size in each mini-batch')
    parser.add_argument('--epochs', type=int, default=40,
                        help='epochs for training')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(3, 1, bilinear=False)
    net.to(device=device)
    data_path_train = args.data_path_train
    Train_Unet(net, device, data_path_train, batch_size=args.batch_size, epochs=args.epochs)
    Unet_train_txt.close()
