import os
import importlib
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from data import create_loader, create_test_loader
from loss import loss as loss_module
from loss.loss import dice_coefficient, loss_fun_gen, logits_to_binary_segmentation
from trainer.common import timer
from model.unet import UNet




class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0
        self.maxdice = 0
        # setup data set and data loader
        self.dataloader, self.dataloader_generator = create_loader(args)
        self.test_dataloader = create_test_loader(args)
        # set up losses and metrics
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        self.dualadv_loss = loss_module.dualadv(args)

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]
        net = importlib.import_module('model.'+args.model)
        
        self.netG = net.InpaintGenerator(args).cuda()
        if args.netD == 'Unet':
            self.netD = net.UnetDiscriminator(args).cuda()
        elif args.netD == 'ResUnet':
            self.netD = net.ResUnetDiscriminator(args).cuda()

        self.netS = UNet(3, 1, bilinear=False).cuda()

        self.contrastive_loss = getattr(loss_module, 'Contrastive')(args, self.netD, args.no_mlp)

        if self.args.use_opt_S:
            if not args.no_mlp:
                self.optimG = torch.optim.Adam(
                    list(self.netG.parameters())+list(self.contrastive_loss.mlp.parameters()),
                    lr=args.lrg, betas=(args.beta1, args.beta2))
            else:
                self.optimG = torch.optim.Adam(
                    list(self.netG.parameters()), lr=args.lrg, betas=(args.beta1, args.beta2))
            self.optimS = torch.optim.Adam(
                self.netS.parameters(), lr=args.lrs, betas=(args.beta1, args.beta2))
        else:
            if not args.no_mlp:
                self.optimG = torch.optim.Adam(
                    list(self.netG.parameters())+list(self.contrastive_loss.mlp.parameters()) +
                    list(self.netS.parameters()),
                    lr=args.lrg, betas=(args.beta1, args.beta2))
            else:
                self.optimG = torch.optim.Adam(
                    list(self.netG.parameters())+list(self.netS.parameters()),
                    lr=args.lrg, betas=(args.beta1, args.beta2))

        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))


        if args.resume:
            self.load()     

        if args.tensorboard: 
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
            

    def load(self):
        try: 
            gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'ckpt', 'G*.pt'))))[-1]
            self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            print(f'[**] Loading generator network from {gpath}')
        except: 
            pass 

        try: 
            dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'ckpt', 'D*.pt'))))[-1]
            data = torch.load(dpath, map_location='cuda')
            if not self.args.no_mlp:
                self.netD.load_state_dict(data['netD'])
                self.contrastive_loss.mlp.load_state_dict(data['MLP'])
                print(f'[**] Loading discriminator and mlp network from {dpath}')
            else:
                self.netD.load_state_dict(data)
                print(f'[**] Loading discriminator network from {dpath}')
        except: 
            pass

        try:
            spath = sorted(list(glob(os.path.join(self.args.save_dir, 'ckpt', 'S*.pt'))))[-1]
            self.netS.load_state_dict(torch.load(spath, map_location='cuda'))
            self.iteration = int(os.path.basename(spath)[1:-3])
            print(f'[**] Loading generator network from {spath}')
        except:
            pass

        try: 
            opath = sorted(list(glob(os.path.join(self.args.save_dir, 'ckpt', 'O*.pt'))))[-1]
            data = torch.load(opath, map_location='cuda')
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            if self.args.use_opt_S:
                self.optimS.load_state_dict(data['optimS'])
            print(f'[**] Loading optimizer from {opath}')
        except: 
            pass



    def save(self, ):
        print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
        torch.save(self.netG.state_dict(),
                os.path.join(self.args.save_dir, 'ckpt', f'G{str(self.iteration).zfill(7)}.pt'))

        torch.save(self.netS.state_dict(),
                os.path.join(self.args.save_dir, 'ckpt', f'S{str(self.iteration).zfill(7)}.pt'))

        if not self.args.no_mlp:
            torch.save(
                {'netD': self.netD.state_dict(), 'MLP': self.contrastive_loss.mlp.state_dict()},
                os.path.join(self.args.save_dir, 'ckpt', f'D{str(self.iteration).zfill(7)}.pt'))
        else:
            torch.save(self.netD.state_dict(),
                os.path.join(self.args.save_dir, 'ckpt', f'D{str(self.iteration).zfill(7)}.pt'))

        if self.args.use_opt_S:
            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict(), 'optimS': self.optimS.state_dict()},
                os.path.join(self.args.save_dir, 'ckpt', f'O{str(self.iteration).zfill(7)}.pt'))
        else:
            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()},
                os.path.join(self.args.save_dir, 'ckpt', f'O{str(self.iteration).zfill(7)}.pt'))
            

    def train(self):
        pbar = tqdm(range(self.args.iterations), initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
        timer_data, timer_model = timer(), timer()


        for idx in pbar:
            self.iteration += 1

            images, masks, filename = next(self.dataloader_generator)
            segs = images[:, 3, :, :].unsqueeze(1)
            images, segs, masks = images.cuda(), segs.cuda(), masks.cuda()
            images_masked = (images * (1 - masks).float()) + masks

            timer_data.hold()
            timer_model.tic()

            pred_img = self.netG(images_masked, masks)
            comp_img = (1 - masks) * images + masks * pred_img

            if self.iteration > self.args.gt_begin_iters:
                img_segs = self.netS(images[:, :3, :, :])
                seg_loss_gt = nn.BCEWithLogitsLoss()(img_segs, segs.to(torch.float32))
                comp_segs = self.netS(comp_img[:, :3, :, :])
                if self.iteration > self.args.seg_begin_iters:
                    seg_loss_comp = loss_fun_gen(comp_segs, segs, masks)
                    seg_loss = seg_loss_gt + 0.1 * seg_loss_comp
                else:
                    seg_loss = seg_loss_gt

            losses = {}
            D_losses = {}
            G_losses = {}

            # optimize D
            D_losses[f"global_advd"], D_losses[f"scat_advd"] = self.dualadv_loss.D(self.netD, comp_img, images, 1-masks)
            self.optimD.zero_grad()
            sum(D_losses.values()).backward()         
            self.optimD.step()

            # optimize G
            # reconstruction loss
            for name, weight in self.args.rec_loss.items(): 
                G_losses[name] = weight * self.rec_loss_func[name](pred_img, images)

            # dual adversarial loss
            G_losses[f"global_advg"], G_losses[f"scat_advg"] = self.dualadv_loss.G(self.netD, comp_img, 1-masks)
            G_losses[f"global_advg"], G_losses[f"scat_advg"] = self.args.adv_weight * G_losses[f"global_advg"], self.args.adv_weight * G_losses[f"scat_advg"]

            # contrastive learning losses
            textural_loss, semantic_loss = self.contrastive_loss(images_masked, comp_img, images, masks, 3)
            G_losses["contra_tex"] = self.args.text_weight * textural_loss
            G_losses["contra_sem"] = self.args.sem_weight * semantic_loss
            if not self.args.use_opt_S and self.iteration > self.args.gt_begin_iters:
                G_losses["seg"] = self.args.seg_weight * seg_loss

            if self.args.use_opt_S and self.iteration > self.args.gt_begin_iters:
                # netS backforward
                self.optimS.zero_grad()
                seg_loss.backward(retain_graph=True)
                self.optimS.step()

            # netG and mlp backforward
            self.optimG.zero_grad()
            sum(G_losses.values()).backward()
            self.optimG.step()

            for name, value in D_losses.items():
                losses[name] = value
            for name, value in G_losses.items():
                losses[name] = value

            if self.iteration > self.args.gt_begin_iters:
                losses["seg"] = seg_loss

            if (self.iteration % self.args.dice_every == 0) and self.args.tensorboard:
                self.netS.eval()
                dices = list()
                for idx, (gt, label) in enumerate(self.test_dataloader):
                    gt = gt.cuda()
                    label = label.cuda()
                    pred_real = self.netS(gt)
                    binary_pred = logits_to_binary_segmentation(pred_real)
                    dice = dice_coefficient(binary_pred, label)
                    dices.append(dice.item())
                mean_dice = torch.mean(torch.tensor(dices)).item()
                if mean_dice > self.maxdice:
                    torch.save(self.netS.state_dict(),
                    os.path.join(self.args.save_dir, 'ckpt', f'Dice_{mean_dice:.5f}_{str(self.iteration).zfill(7)}.pt'))
                    self.maxdice = mean_dice
                print(f'--------------------------DICE {self.iteration}-----------------------')
                print(f'dice={mean_dice}')
                print(f'--------------------------DICE {self.iteration}-----------------------')
                self.writer.add_scalar('dice', mean_dice, self.iteration)

            # logs
            if self.iteration % self.args.print_every == 0:
                pbar.update(self.args.print_every)
                description = f'epoch:{self.iteration}, mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                for key, val in losses.items(): 
                    description += f'{key}:{val.item():.3f}, '
                    if self.args.tensorboard and (self.iteration % self.args.writelog_every == 0):
                        self.writer.add_scalar(key, val.item(), self.iteration)
                pbar.set_description((description))

            if self.iteration % self.args.plot_every == 0 and self.args.tensorboard:
                self.writer.add_image('mask', make_grid(masks), self.iteration)
                self.writer.add_image('orig', make_grid((images[:, :3, :, :]+1.0)/2.0), self.iteration)
                self.writer.add_image('comp', make_grid((comp_img[:, :3, :, :]+1.0)/2.0), self.iteration)
                self.writer.add_image('segorig', make_grid((segs > 0).to(torch.float32)), self.iteration)
                if self.iteration > self.args.gt_begin_iters:
                    self.writer.add_image('segcomp', make_grid((logits_to_binary_segmentation(comp_segs)).to(torch.float32)), self.iteration)

            
            if self.iteration % self.args.save_every == 0:
                self.save()