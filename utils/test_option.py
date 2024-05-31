import argparse

parser = argparse.ArgumentParser(description='Image Inpainting')

# data specifications
parser.add_argument('--dir_image', type=str, default='/home/pc/diskB/jjm/cell/dataset/train/imgs/',
                    help='image dataset directory for training or testing')
parser.add_argument('--dataset', type=str, default='MoNuSeg',
                    help='training dataset: (Places2 | CelebA)')
parser.add_argument('--image_size', type=int, default=256,
                    help='image size used during training')
parser.add_argument('--crop_size', type=int, default=256,
                    help='image crop size used during training, use 178 for CelebA dataset')
parser.add_argument('--transform', type=str, default='randomcrop',
                    help='image transformation type: (randomcrop | centercrop | resize_and_crop), use centercrop and randomcrop for CelebA and Places2 respectively')
parser.add_argument('--mask_type', type=str, default='random',
                    help='mask used during training: (centered | random), pconv needs to specify --dir_mask')


# model specifications
parser.add_argument('--model', type=str, default='model',
                    help='model name')
parser.add_argument('--block_num', type=int, default=8,
                    help='number of AOT blocks')
parser.add_argument('--rates', type=str, default='1+2+4+8',
                    help='dilation rates used in AOT block')
parser.add_argument('--netD', type=str, default='Unet',
                    help='discriminator network: (Unet | ResUnet)')
parser.add_argument('--use_D_attn', action='store_true',
                    help='use self-attention in netD')
parser.add_argument('--no_SN', action='store_true',
                    help='not use spectral normalization in netD')
parser.add_argument('--globalgan_type', type=str, default='hingegan',
                    help='global adversarial training: (hingegan | nsgan | lsgan)')
parser.add_argument('--SCAT_type', type=str, default='hingegan',
                    help='segmentation confusion adversarial training: (hingegan | nsgan | lsgan)')
parser.add_argument('--no_mlp', action='store_true',
                    help='use mlp for semantic contrastive loss')

# hardware specifications
parser.add_argument('--seed', type=int, default=77,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers used in data loader')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size in each mini-batch')



# test specifications
parser.add_argument('--pre_train', type=str, default='./ckpt/Generative.pt',
                    help='path to pretrained models')
parser.add_argument('--pre_train_seg', type=str, default='./ckpt/Plugged_UNet.pt',
                    help='path to pretrained models')
parser.add_argument('--outputs', type=str, default='../outputs/outputs_256new',
                    help='path to save results')
parser.add_argument('--num_test',  type=int, default=500,
                    help='required number of images to generate')
parser.add_argument('--file_name',  type=str, default=None,
                    help='the specific type of image for generating')
parser.add_argument('--shuffle', default=True,
                    help='sample random images for testing')


# ----------------------------------
args = parser.parse_args()
args.rates = list(map(int, list(args.rates.split('+'))))