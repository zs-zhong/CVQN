import torch
import torch.utils.data as data
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os.path as osp
import time
import pdb

from pytorch_msssim import msssim
from config import config
from config import update_config, create_logger
import dataset
from component import *

class Trainer():
    def __init__(self, cfg, logger, model_dir, tensorboard_log_dir, **kwargs):
        torch.cuda.set_device(cfg['GPU_DEVICE'])

        self.nums_epoch = cfg['TRAIN']['NUM_EPOCH']
        self.epoch = 0
        self.wr_train_idx = 1
        self.wr_eval_idx = 1
        self.scale = 16
        self.max_psnr = 0.
        self.max_ssim = 0.
        self.its_bpp = 24.

        self.log = logger
        self.alpha = cfg['TRAIN']['ALPHA']
        self.beta = cfg['TRAIN']['BETA']
        self.print_freq = cfg['PRINT_FREQ']
        self.quant_level = cfg['QUA_LEVELS']

        self.model_dir = model_dir
        self.writer = SummaryWriter(tensorboard_log_dir)

        self.encoder = Encoder(cfg).cuda()
        self.decoder = Decoder(cfg).cuda()

        self.quantizer = nn.ModuleList()
        self.entropy = nn.ModuleList()

        self.clip_value = cfg['TRAIN']['CLIP_VALUE'] 


        self.part = [0]
        [self.part.append(int(x*cfg['CODE_CHNS']) + self.part[i]) for i, x in enumerate(cfg['ENP']['PART'])]

        for i in range(len(cfg['ENP']['PART'])):
            self.quantizer.append(GMMQuantizer(cfg['QUA_LEVELS'][i], cfg['QUA']['STD'][i], cfg['QUA']['PI'][i]).cuda())
            self.entropy.append(ContextEstimater3d(cfg['QUA_LEVELS'][i], cfg['ENP']['FEAT_NUMS'][i]).cuda())
             

        self.code_nums = cfg['CODE_CHNS']
        self.solver_enc = optim.Adam(self.encoder.parameters(), lr=cfg['TRAIN']['LR_ENC'])

        Q_params = list()
        for i in range(len(cfg['QUA_LEVELS'])):
            Q_params.append({'params': self.quantizer[i].mean,    'lr': cfg['TRAIN']['LR_QUA']      })
            Q_params.append({'params': self.quantizer[i].log_pi,  'lr': cfg['TRAIN']['LR_QUA'] * 0.2})
            Q_params.append({'params': self.quantizer[i].log_std, 'lr': cfg['TRAIN']['LR_QUA'] * 0.2})
        self.solver_qua = optim.Adam(Q_params)


        self.solver_etp = optim.Adam(self.entropy.parameters(), lr=cfg['TRAIN']['LR_ENP'])
        self.solver_dec = optim.Adam(self.decoder.parameters(), lr=cfg['TRAIN']['LR_DEC'])


        self.scheduler_enc = LS.MultiStepLR(self.solver_enc, milestones=cfg['TRAIN']['MILESTONES'], gamma=cfg['TRAIN']['GAMMA'])
        self.scheduler_qua = LS.MultiStepLR(self.solver_qua, milestones=cfg['TRAIN']['MILESTONES'], gamma=cfg['TRAIN']['GAMMA'])
        self.scheduler_etp = LS.MultiStepLR(self.solver_etp, milestones=cfg['TRAIN']['MILESTONES'], gamma=cfg['TRAIN']['GAMMA'])
        self.scheduler_dec = LS.MultiStepLR(self.solver_dec, milestones=cfg['TRAIN']['MILESTONES'], gamma=cfg['TRAIN']['GAMMA'])


        train_transform = transforms.Compose([
            transforms.RandomCrop((cfg['DATASET']['PATCH_SIZE'], cfg['DATASET']['PATCH_SIZE'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),])
        train_set = dataset.ImageFolder(root=cfg['DATASET']['TRAIN_SET'], transform=train_transform)
        self.train_loader = data.DataLoader(dataset=train_set, 
            batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=cfg['WORKERS'])

        test_transform = transforms.Compose([
            transforms.ToTensor(),])
        test_set = dataset.ImageFolder(root=cfg['DATASET']['TEST_SET'], transform=test_transform)
        self.test_loader = data.DataLoader(dataset=test_set, 
            batch_size=1, shuffle=False, num_workers=cfg['WORKERS'])

        self.entropy_loss = nn.CrossEntropyLoss()



    def train(self):
        self.log.info('------------------------------Training------------------------------')


        self._train_init()
        
        for batch, (img, _) in enumerate(self.train_loader):
            
            batch_t0 = time.time()


            self._solver_init()
            
            # float image 0~1
            img = img.cuda()
            
            # feat: 32 x 64 x 8 x 8      
            feat = self.encoder(img)
            # bc: 32 x 64 x 8 x 8 (int)

            feat_new = self._order_feat(feat)
            
            bc_list = list()
            idx_list = list()
            prob_list = list()
            # loss_gmm = 0
            for i in range(len(self.part) - 1):
                
                bc, idx = self.quantizer[i](feat_new[:, self.part[i]:self.part[i + 1], :, :])
                # loss_gmm -= torch.mean(torch.log(self.quantizer[i].get_prob(feat_new[:,self.part[i]:self.part[i+1],:,:])))
                bc_list.append(bc)
                idx_list.append(idx.squeeze(-1))
                prob_list.append(self.entropy[i](bc_list[i].unsqueeze(1)))

            # loss_gmm = loss_gmm / (len(self.part)-1)

            bc = bc_list[0]
            for i in range(1, len(self.part) - 1):
                bc = torch.cat((bc, bc_list[i]), dim=1)


            bc_reorder = self._reorder_code(bc)

            re = self.decoder(bc_reorder)

            img255 = img.mul(255).add_(0.5).clamp_(0, 255).floor()
            re_ = re.mul(255).add_(0.5).clamp_(0, 255)
            re255 = (re_.floor() - re_).detach() + re_
            
            loss_distortion = 1 - msssim(img255, re255)


            loss_entropy = 0

            for i in range(len(idx_list)):
                loss_entropy += self.entropy_loss(prob_list[i], idx_list[i])

            loss_entropy = loss_entropy / len(idx_list)

            # loss = self.alpha * loss_distortion + loss_entropy + self.beta * loss_gmm
            loss = self.alpha * loss_distortion + loss_entropy

            loss.backward()

            self._solver_update()

            if (batch + 1) % (self.print_freq) == 0:

                self.writer.add_scalar('loss_dis', loss_distortion, self.wr_train_idx)
                self.writer.add_scalar('loss_en', loss_entropy, self.wr_train_idx)
                self.wr_train_idx += 1

                batch_t1 = time.time()
                for i in range(len(self.part) - 1):
                    self.log.info('Quantizer[{}/{}]: '.format(i + 1, len(self.part) - 1) + ' mean:\t' + str(np.round(self.quantizer[i].mean.data.cpu().numpy(),4)))
                    self.log.info('Quantizer[{}/{}]: '.format(i + 1, len(self.part) - 1) + ' std:\t'  + str(np.round(self.quantizer[i].std.data.cpu().numpy(),4)))
                    self.log.info('Quantizer[{}/{}]: '.format(i + 1, len(self.part) - 1) + ' pi:\t'   + str(np.round(self.quantizer[i].norm_pi.data.cpu().numpy(),4)))
                self.log.info('Epoch[{}/{}]({}/{}):\t'.format(self.epoch + 1, self.nums_epoch, batch + 1, len(self.train_loader)) \
                     + '\t Loss_dis: {:.3f};\t Loss_etp: {:.3f};\t Batch time: {:.3f} sec.'
                    .format(loss_distortion.item(), loss_entropy.item(), batch_t1 - batch_t0))
             

        self.epoch += 1

    def eval(self):
        self.log.info('-----------------------------Evaluation-----------------------------')
        eval_t0 = time.time()

        self._eval_init()
        

        ssim_mean = []
        psnr_mean = []
        bpp_mean = []

        for _, (img, _) in enumerate(self.test_loader):
            
            # float image 0~1
            img = img.cuda()
            _, _, h, w = img.size()
            
            # feat: 32 x 64 x 8 x 8      
            feat = self.encoder(img)
            # bc: 32 x 64 x 8 x 8 (int)

            feat_new = self._order_feat(feat)
            
            
            bc_list = list()
            idx_list = list()
            prob_list = list()
            for i in range(len(self.part) - 1):
                bc, idx = self.quantizer[i](feat_new[:,self.part[i]:self.part[i + 1],:,:])
                bc_list.append(bc)
                idx_list.append(idx.squeeze(-1))
                prob_list.append(self.entropy[i](bc_list[i].unsqueeze(1)))

            bc = bc_list[0]
            for i in range(1, len(self.part) - 1):
                bc = torch.cat((bc, bc_list[i]), dim=1)
            

            bc_reorder = self._reorder_code(bc)

            re = self.decoder(bc_reorder)
            
            bpp_item = 0
            for i in range(len(idx_list)):
                bpp_item += F.cross_entropy(prob_list[i], idx_list[i], reduction='sum')

        
            bpp_item = bpp_item * torch.log2(torch.tensor(2.71828)).cuda() / h / w
            bpp_mean.append(bpp_item.item())

            img255 = img.squeeze().mul(255).add_(0.5).clamp_(0, 255).floor()

            re255 = re.squeeze().mul(255).add_(0.5).clamp_(0, 255).floor()

            psnr_item = self.compute_psnr(img255.unsqueeze(0), re255.unsqueeze(0))

            psnr_mean.append(psnr_item.item())
            
            ssim_item = msssim(img255.unsqueeze(0), re255.unsqueeze(0))

            ssim_mean.append(ssim_item.item())
            
            
        eval_t1 = time.time()

        self.writer.add_scalar('BPP', np.mean(bpp_mean), self.wr_eval_idx)
        self.writer.add_scalar('MS-SSIM', np.mean(ssim_mean), self.wr_eval_idx)
        
        self.wr_eval_idx += 1

        self.log.info('Mean PSNR: {:.3f};\t Mean MS-SSIM: {:.5f};\t Mean BPP: {:.5f};\t Test time: {:.3f} sec.'.
            format(np.mean(psnr_mean), np.mean(ssim_mean), np.mean(bpp_mean), eval_t1 - eval_t0))
        if  np.mean(ssim_mean) > self.max_ssim:
            self.save_checkpoint('best.pth')
            self.max_psnr = np.mean(psnr_mean)
            self.max_ssim = np.mean(ssim_mean)
            self.its_bpp = np.mean(bpp_mean)
        self.log.info('Best PSNR: {:.3f};\t Best MS-SSIM: {:.5f};\t Its BPP: {:.5f}.'.
            format(self.max_psnr, self.max_ssim, self.its_bpp))
        

    def save_checkpoint(self, model_name):
        raise NotImplementedError("Must inherit from Trainer.")

    def load_checkpoint(self, date, model_name):
        raise NotImplementedError("Must inherit from Trainer.")

    def update_lr(self):
        raise NotImplementedError("Must inherit from Trainer.")

    def _train_init(self):
        raise NotImplementedError("Must inherit from Trainer.")

    def _eval_init(self):
        raise NotImplementedError("Must inherit from Trainer.")

    def _solver_init(self):
        raise NotImplementedError("Must inherit from Trainer.")

    def _solver_update(self):
        raise NotImplementedError("Must inherit from Trainer.")

    def _order_feat(self, feat):
        raise NotImplementedError("Must inherit from Trainer.")

    def _reorder_code(self, bc):
        raise NotImplementedError("Must inherit from Trainer.")
    

    def compute_psnr(self, img, re):
        img = img.squeeze()
        re = re.squeeze()
        mse = torch.mean((img - re) ** 2) 
        psnr = 10 * (2 * torch.log10(torch.tensor(255.).cuda()) - torch.log10(mse))
        return psnr

    def rgb2yCbCr(self, img_rgb):
        im_flat = img_rgb.contiguous().view(-1, 3).float()
        mat = torch.tensor([[65.481, 128.553, 24.966],
            [-37.797, -74.203, 112.0],
            [112.0, -93.786, -18.214]]).cuda()
        bias = torch.tensor([16.0, 128.0, 128.0]).cuda()
        img_yCbCr = torch.round(im_flat.mm(mat.T) * 1.0 / 255  + bias)
        img_yCbCr = img_yCbCr.view(input_im.shape[0], input_im.shape[1], 3)
        return img_yCbCr

# class PlainTrainer(Trainer):
#     def __init__(self, cfg, logger, model_dir, tensorboard_log_dir, **kwargs):
#         super(PlainTrainer, self).__init__(cfg, logger, model_dir, tensorboard_log_dir, **kwargs)




class PredefineTrainer(Trainer):
    def __init__(self, cfg, logger, model_dir, tensorboard_log_dir, **kwargs):
        super(PredefineTrainer, self).__init__(cfg, logger, model_dir, tensorboard_log_dir, **kwargs)


    def save_checkpoint(self, model_name):
        self.log.info('------------------------------Save Cpt------------------------------')
        
        ckp_path = osp.join(self.model_dir, model_name)

        self.log.info('Save checkpoint: %s' % ckp_path)
        
        obj = {
            'encoder': self.encoder.state_dict(),
            'quantizer': self.quantizer.state_dict(),
            'decoder': self.decoder.state_dict(),
            'entropy': self.entropy.state_dict(),
            'epoch': self.epoch
        }
        torch.save(obj, ckp_path)
        self.log.info('Save the trained model successfully.')

    def load_checkpoint(self, date, model_name):
        self.log.info('------------------------------Load Cpt------------------------------')
        ckp_path = osp.join(self.model_dir[:-12] + date, model_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('Load checkpoint: %s' % ckp_path)
        except IOError:
            self.log.error('No checkpoint: %s!' % ckp_path)
            return
        self.encoder.load_state_dict(obj['encoder'])
        self.quantizer.load_state_dict(obj['quantizer'])
        self.decoder.load_state_dict(obj['decoder'])
        self.entropy.load_state_dict(obj['entropy'])
        self.epoch = obj['epoch']
        self.log.info('The loaded model has been trained for %d epoch(s).' % self.epoch)


    def update_lr(self):
        self.scheduler_enc.step()
        self.scheduler_qua.step()
        self.scheduler_dec.step()
        self.scheduler_etp.step()

    def _train_init(self):
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.quantizer.train()
        for param in self.quantizer.parameters():
            param.requires_grad = True

        self.entropy.train()
        for param in self.entropy.parameters():
            param.requires_grad = True

        self.decoder.train()
        for param in self.decoder.parameters():
            param.requires_grad = True


    def _eval_init(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.quantizer.eval()
        for param in self.quantizer.parameters():
            param.requires_grad = False

        self.entropy.eval()
        for param in self.entropy.parameters():
            param.requires_grad = False

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def _solver_init(self):
        self.solver_qua.zero_grad()
        self.solver_dec.zero_grad()
        self.solver_etp.zero_grad()
        self.solver_enc.zero_grad()

    def _solver_update(self):
        torch.nn.utils.clip_grad_value_(self.encoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.decoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.quantizer.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.entropy.parameters(), self.clip_value)

        self.solver_enc.step()
        self.solver_qua.step()
        self.solver_dec.step()
        self.solver_etp.step()

    def _order_feat(self, feat):
        return feat

    def _reorder_code(self, bc):
        return bc


class RETrainer(Trainer):
    def __init__(self, cfg, logger, model_dir, tensorboard_log_dir, **kwargs):
        super(RETrainer, self).__init__(cfg, logger, model_dir, tensorboard_log_dir, **kwargs)
        self.imp_value = torch.ones(cfg['CODE_CHNS']).cuda()
        self.order = torch.randperm(cfg['CODE_CHNS']).cuda()
        _, self.reorder = torch.sort(self.order)
        self.log.info('initialized imp order: ')
        self.log.info(self.order)
        imp_transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),])
        imp_set = dataset.ImageFolder(root=cfg['DATASET']['TRAIN_SET'], transform=imp_transform)
        self.imp_loader = data.DataLoader(dataset=imp_set, batch_size=1, shuffle=True, num_workers=cfg['WORKERS'])
        self.end_batch = cfg['RE_END_BATCH']


    def save_checkpoint(self, model_name):
        self.log.info('------------------------------Save Cpt------------------------------')
        
        ckp_path = osp.join(self.model_dir, model_name)

        self.log.info('Save checkpoint: %s' % ckp_path)
        
        obj = {
            'encoder': self.encoder.state_dict(),
            'quantizer': self.quantizer.state_dict(),
            'decoder': self.decoder.state_dict(),
            'entropy': self.entropy.state_dict(),
            'epoch': self.epoch
        }
        torch.save(obj, ckp_path)
        self.log.info('Save the trained model successfully.')

    def load_checkpoint(self, date, model_name):
        self.log.info('------------------------------Load Cpt------------------------------')
        ckp_path = osp.join(self.model_dir[:-12] + date, model_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('Load checkpoint: %s' % ckp_path)
        except IOError:
            self.log.error('No checkpoint: %s!' % ckp_path)
            return
        self.encoder.load_state_dict(obj['encoder'])
        self.quantizer.load_state_dict(obj['quantizer'])
        self.decoder.load_state_dict(obj['decoder'])
        self.entropy.load_state_dict(obj['entropy'])
        self.epoch = obj['epoch']
        self.log.info('The loaded model has been trained for %d epoch(s).' % self.epoch)


    def update_lr(self):
        self.scheduler_enc.step()
        self.scheduler_qua.step()
        self.scheduler_dec.step()
        self.scheduler_etp.step()

    def _train_init(self):
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.quantizer.train()
        for param in self.quantizer.parameters():
            param.requires_grad = True

        self.entropy.train()
        for param in self.entropy.parameters():
            param.requires_grad = True

        self.decoder.train()
        for param in self.decoder.parameters():
            param.requires_grad = True


    def _eval_init(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.quantizer.eval()
        for param in self.quantizer.parameters():
            param.requires_grad = False

        self.entropy.eval()
        for param in self.entropy.parameters():
            param.requires_grad = False

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def _solver_init(self):
        self.solver_qua.zero_grad()
        self.solver_dec.zero_grad()
        self.solver_etp.zero_grad()
        self.solver_enc.zero_grad()

    def _solver_update(self):
        torch.nn.utils.clip_grad_value_(self.encoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.decoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.quantizer.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.entropy.parameters(), self.clip_value)

        self.solver_enc.step()
        self.solver_qua.step()
        self.solver_dec.step()
        self.solver_etp.step()

    def _order_feat(self, feat):
        return feat[:, self.order, :, :] 

    def _reorder_code(self, bc):
        return bc[:,self.reorder,:,:]

    def re_based_get_imp(self):
        self.log.info('-----------------Get Importance Based on RE------------------')
        imp_t0 = time.time()

        self._eval_init()

        ssim_mean = torch.zeros(self.code_nums, self.end_batch)
        l2_err_mean = torch.zeros(self.code_nums, self.end_batch)
        bpp_mean = torch.zeros(self.code_nums, self.end_batch)
        
        for batch, (img, _) in enumerate(self.imp_loader):
            if batch == self.end_batch:
                break

            # float image 0~1
            img = img.cuda()
            
            # feat: 32 x 64 x 8 x 8      
            feat = self.encoder(img)
            # bc: 32 x 64 x 8 x 8 (int)
            feat_new = self._order_feat(feat)

            
            bc_list = list()
            idx_list = list()
            prob_list = list()
            for i in range(len(self.part)-1):
                bc, idx = self.quantizer[i](feat_new[:,self.part[i]:self.part[i+1],:,:])
                bc_list.append(bc)
                idx_list.append(idx.squeeze(-1))
                prob_list.append(self.entropy[i](bc_list[i].unsqueeze(1)))

            bc = bc_list[0]
            for i in range(1, len(self.part)-1):
                bc = torch.cat((bc, bc_list[i]), dim=1)
            

            bc_reorder = self._reorder_code(bc)

            re = self.decoder(bc_reorder)

            _, ch, _, _ = bc_reorder.size()
            _, _, h, w = img.size()

            for i in range(ch):
                bc_reorder_new = bc_reorder.clone()
                bc_reorder_new[:,i,:,:] = 0
                re_ch = self.decoder(bc_reorder_new)
                ssim_mean[i, batch] = msssim(re, re_ch)
                l2_err_mean[i, batch] = torch.mean((re-re_ch)**2)
        
        ssim_mean = torch.mean(ssim_mean, dim=1)
        l2_err_mean = torch.mean(l2_err_mean, dim=1)
        bpp_mean = torch.mean(bpp_mean, dim=1)
        bpp_mean = bpp_mean[self.reorder]
        self.imp_value = (1 - ssim_mean) * 200
        self.log.info("Current importance value: ")
        self.log.info(self.imp_value)
        _, self.order = torch.sort(self.imp_value)
        self.log.info("Current importance order: ")
        self.log.info(self.order)
        _, self.reorder = torch.sort(self.order)
        imp_t1 = time.time()
        self.log.info('Update channel importance time: {:.3f} sec.'.format(imp_t1 - imp_t0))


class SETrainer(Trainer):
    def __init__(self, cfg, logger, model_dir, tensorboard_log_dir, **kwargs):
        super(SETrainer, self).__init__(cfg, logger, model_dir, tensorboard_log_dir, **kwargs)
        self.se_based = ChannelImportance(cfg['CODE_CHNS']).cuda()
        self.solver_se = optim.Adam(self.se_based.parameters(), lr=cfg['LR_SE'])
        self.scheduler_se = LS.MultiStepLR(self.solver_se, milestones=cfg['TRAIN']['MILESTONES'], gamma=cfg['TRAIN']['GAMMA'])
        self.order = torch.randperm(cfg['CODE_CHNS']).cuda()
        _, self.reorder = torch.sort(self.order)

    def save_checkpoint(self, model_name):
        self.log.info('------------------------------Save Cpt------------------------------')
        
        ckp_path = osp.join(self.model_dir, model_name)

        self.log.info('Save checkpoint: %s' % ckp_path)
        
        obj = {
            'encoder': self.encoder.state_dict(),
            'quantizer': self.quantizer.state_dict(),
            'decoder': self.decoder.state_dict(),
            'entropy': self.entropy.state_dict(),
            'chn_imp': self.se_based.state_dict(),
            'epoch': self.epoch
        }
        torch.save(obj, ckp_path)
        self.log.info('Save the trained model successfully.')

    def load_checkpoint(self, date, model_name):
        self.log.info('------------------------------Load Cpt------------------------------')
        ckp_path = osp.join(self.model_dir[:-12] + date, model_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('Load checkpoint: %s' % ckp_path)
        except IOError:
            self.log.error('No checkpoint: %s!' % ckp_path)
            return
        self.encoder.load_state_dict(obj['encoder'])
        self.quantizer.load_state_dict(obj['quantizer'])
        self.decoder.load_state_dict(obj['decoder'])
        self.entropy.load_state_dict(obj['entropy'])
        self.se_based.load_state_dict(obj['chn_imp'])
        self.epoch = obj['epoch']
        self.log.info('The loaded model has been trained for %d epoch(s).' % self.epoch)


    def update_lr(self):
        self.scheduler_enc.step()
        self.scheduler_qua.step()
        self.scheduler_dec.step()
        self.scheduler_etp.step()
        self.scheduler_se.step()

    def _train_init(self):
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.quantizer.train()
        for param in self.quantizer.parameters():
            param.requires_grad = True

        self.entropy.train()
        for param in self.entropy.parameters():
            param.requires_grad = True

        self.decoder.train()
        for param in self.decoder.parameters():
            param.requires_grad = True

        self.se_based.train()
        for param in self.se_based.parameters():
            param.requires_grad = True


    def _eval_init(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.quantizer.eval()
        for param in self.quantizer.parameters():
            param.requires_grad = False

        self.entropy.eval()
        for param in self.entropy.parameters():
            param.requires_grad = False

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.se_based.train()
        for param in self.se_based.parameters():
            param.requires_grad = False

    def _solver_init(self):
        self.solver_qua.zero_grad()
        self.solver_dec.zero_grad()
        self.solver_etp.zero_grad()
        self.solver_enc.zero_grad()
        self.solver_se.zero_grad()

    def _solver_update(self):
        torch.nn.utils.clip_grad_value_(self.encoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.decoder.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.quantizer.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.entropy.parameters(), self.clip_value)
        torch.nn.utils.clip_grad_value_(self.se_based.parameters(), self.clip_value)

        self.solver_enc.step()
        self.solver_qua.step()
        self.solver_dec.step()
        self.solver_etp.step()
        self.solver_se.step()

    def _order_feat(self, feat):
        imp_value = self.se_based(feat)
        mean_imp_value = torch.mean(imp_value, axis=0)
        _, self.order = torch.sort(mean_imp_value.squeeze())
        _, self.reorder = torch.sort(self.order)
        return feat[:, self.order, :, :]

    def _reorder_code(self, bc):
        return bc[:,self.reorder,:,:]

