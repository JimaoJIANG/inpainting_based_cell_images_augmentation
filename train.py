import os
import torch
from utils.option import args
from trainer.trainer import Trainer


def main_worker(args):
    args.save_dir = os.path.join(
        args.save_dir, f'{args.model}_{args.dataset}_{args.mask_type}{args.image_size}')

    os.makedirs(os.path.join(args.save_dir, 'ckpt'), exist_ok=True)
    with open(os.path.join(args.save_dir, 'config.txt'), 'a') as f:
        for key, val in vars(args).items():
            f.write(f'{key}: {val}\n')
    print(f'[**] create folder {args.save_dir}')

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker(args)
