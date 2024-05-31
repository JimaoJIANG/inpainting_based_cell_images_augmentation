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

# optimization specifications 
parser.add_argument('--lrg', type=float, default=1e-4,
                    help='learning rate for generator')
parser.add_argument('--lrd', type=float, default=1e-4,
                    help='learning rate for discriminator')
parser.add_argument('--lrs', type=float, default=1e-4,
                    help='learning rate for discriminator')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--beta1', type=float, default=0,
                    help='beta1 in optimizer')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='beta2 in optimier')

# loss specifications 
parser.add_argument('--rec_loss', type=str, default='100*L1',
                    help='losses for reconstruction')                    
parser.add_argument('--adv_weight', type=float, default=1,
                    help='loss weight for adversarial losses')          
parser.add_argument('--text_weight', type=float, default=10,
                    help='loss weight for textural contrastive loss')  
parser.add_argument('--sem_weight', type=float, default=1,
                    help='loss weight for semantic contrastive loss')            
parser.add_argument('--seg_weight', type=float, default=1,
                    help='loss weight for segmentation')

# training specifications 
parser.add_argument('--iterations', type=int, default=3e5,
                    help='the number of iterations for training')
parser.add_argument('--gt_begin_iters', type=int, default=1e5,
                    help='the number of iterations for training')
parser.add_argument('--seg_begin_iters', type=int, default=1e5 + 2e4,
                    help='the number of iterations for training')
parser.add_argument('--use_opt_S', type=bool, default=True,
                    help='use separate optimizer for segmentation')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size in each mini-batch')
parser.add_argument('--resume', type=bool, default=True,
                    help='resume from previous iteration')

# log specifications 
parser.add_argument('--print_every', type=int, default=100,
                    help='frequency for updating progress bar')
parser.add_argument('--save_every', type=int, default=1e4,
                    help='frequency for saving models')
parser.add_argument('--plot_every', type=int, default=1e3,
                    help='frequency for plot models')
parser.add_argument('--dice_every', type=int, default=500,
                    help='frequency for calculate dice')
parser.add_argument('--writelog_every', type=int, default=200,
                    help='frequency for log losses')
parser.add_argument('--save_dir', type=str, default='../experiments',
                    help='directory for saving models and logs')
parser.add_argument('--tensorboard', default='True',
                    help='default: false, since it will slow training. use it for debugging')

# dice test
parser.add_argument('--test_path', type=str, default=r"/home/pc/diskB/jjm/cell/dataset/test/imgs/",
                    help='path to test data')
parser.add_argument('--test_img_size', type=int, default=1000,
                    help='test image size')


# ----------------------------------
args = parser.parse_args()
args.iterations = int(args.iterations)

args.rates = list(map(int, list(args.rates.split('+'))))

losses = list(args.rec_loss.split('+'))
args.rec_loss = {}
for l in losses: 
    weight, name = l.split('*')
    args.rec_loss[name] = float(weight)