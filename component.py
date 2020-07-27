import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
import math
import pdb


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


## Inverse Pixel Shuffle (IPS, part of encoder/decoder)
class InversePixelShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(InversePixelShuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)


class LowerBound(Function):

    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
    

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


## Generalized Divisive Normalization (GDN, part of encoder/decoder)
class GDN(nn.Module):
    """
    Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
  
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        device = torch.cuda.current_device()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])

        self.build(ch, device)
  
    def build(self, ch, device):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset
  
        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta.to(device))

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma.to(device))
        self.pedestal = self.pedestal.to(device)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma  = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs

## Channel Attention (CA, part of encoder/decoder)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB, part of encoder/decoder)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction=1,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG, part of encoder/decoder)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, n_blocks, kernel_size, reduction=1, act=nn.ReLU(True), res_scale=1):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_blocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## default convolution (part of encoder/decoder)
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## Mean Shift (MS, part of encoder/decoder)
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        # self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


## Encoder
class Encoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Encoder, self).__init__()
        
        if cfg['ENC']['ACT'] == 'relu':
            act = nn.ReLU(True)
        elif cfg['ENC']['ACT'] == 'lrelu': 
            act = nn.LeakyReLU(0.2, True)
        elif cfg['ENC']['ACT'] == 'identity': 
            act = nn.Identity()

        conv = default_conv

        feat_num = cfg['ENC']['FEAT_NUMS']
        
        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_mean = (0.45659249, 0.43772669, 0.41186953)
        rgb_mean = cfg['DATASET']['MEAN']
        rgb_std = cfg['DATASET']['STD']

        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(cfg['IN_CHNS'], feat_num[0], 3)]

        # define body module, four stages
        modules_body = []
        for i in range(4):
            modules_body.append(InversePixelShuffle(2))
            if cfg['ENC']['GDN_FLAG'] == True:
                modules_body.append(GDN(feat_num[i] * 4))
            modules_body.append(ResidualGroup(conv, feat_num[i] * 4, cfg['ENC']['BLOCK_NUMS'][i], 3, act=act))
            if i < 3:
                modules_body.append(conv(feat_num[i] * 4, feat_num[i + 1], 3))
            else :
                modules_body.append(conv(feat_num[i] * 4, feat_num[i], 3))
                           

        # define tail module
        modules_tail = [conv(feat_num[-1], cfg['CODE_CHNS'], 3)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)

        x = self.tail(res)


        return x




## Decoder
class Decoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Decoder, self).__init__()
        conv = default_conv

        if cfg['DEC']['ACT'] == 'relu':
            act = nn.ReLU(True)
        elif cfg['DEC']['ACT'] == 'lrelu': 
            act = nn.LeakyReLU(0.05, True)
        elif cfg['DEC']['ACT'] == 'identity': 
            act = nn.Identity()
        
        feat_num = cfg['DEC']['FEAT_NUMS']
        
        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        
        rgb_mean = cfg['DATASET']['MEAN']
        rgb_std = cfg['DATASET']['STD']
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)
        
        # define head module
        modules_head = [conv(cfg['CODE_CHNS'], feat_num[0], 3)]


        # define body module
        modules_body = []
        for i in range(4):
            if i < 3:
                modules_body.append(conv(feat_num[i], feat_num[i + 1] * 4, 3))
                modules_body.append(ResidualGroup(conv, feat_num[i + 1] * 4, cfg['DEC']['BLOCK_NUMS'][i], 3, act=act))
                if cfg['DEC']['GDN_FLAG'] == True:
                    modules_body.append(GDN(feat_num[i + 1] * 4))
            else :
                modules_body.append(conv(feat_num[i], feat_num[i] * 4, 3))
                modules_body.append(ResidualGroup(conv, feat_num[i] * 4, cfg['DEC']['BLOCK_NUMS'][i], 3, act=act))
                if cfg['DEC']['GDN_FLAG'] == True:
                    modules_body.append(GDN(feat_num[i] * 4))
            
            modules_body.append(nn.PixelShuffle(2))
                
        # define tail module
        modules_tail = [conv(feat_num[-1], cfg['IN_CHNS'], 3)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        x = self.head(x)

        res = self.body(x)

        x = self.tail(res)

        x = self.add_mean(x)

        return x


## 3D Masked Convolution (part of entropy model)
class MaskedConv3d(torch.nn.Conv3d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
        assert mask_type in {'hollow', 'solid'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kZ, kH, kW = self.weight.size()
        self.mask.fill_(1)
        center_idx = kZ * kH * kW // 2
        cur_idx = 0
        for i in range(kZ):
            for j in range(kH):
                for k in range(kW):
                    if cur_idx < center_idx + (mask_type == 'solid'):
                        self.mask[:, :, i, j, k] = 1
                    else:
                        self.mask[:, :, i, j, k] = 0
                    cur_idx += 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)[:, :, :, :, :]


## 3D Entropy Context Estimate Model
class ContextEstimater3d(torch.nn.Module):
    def __init__(self, fm_centers_n, internal_dps):
        super(ContextEstimater3d, self).__init__()
        self.fm_centers_n = fm_centers_n
        self.kernel_n = (3, 3, 3)
        # self.pad = torch.nn.ConstantPad3d((1, 1, 1, 1, 1, 1), 1)
        self.pad_n = (1, 1, 1)
        self.stride_n = (1, 1, 1)
        self.internal_dps = internal_dps
        self.conv_bias = True
        self.active_func = torch.nn.functional.relu

        self.first_layer = MaskedConv3d('hollow', 1, self.internal_dps,
                                        self.kernel_n, self.stride_n, self.pad_n,
                                        bias=self.conv_bias)

        self.middle_layer1 = MaskedConv3d('solid', self.internal_dps, self.internal_dps,
                                          self.kernel_n, self.stride_n, self.pad_n,
                                          bias=self.conv_bias)
        self.middle_layer2 = MaskedConv3d('solid', self.internal_dps, self.internal_dps,
                                          self.kernel_n, self.stride_n, self.pad_n,
                                          bias=self.conv_bias)

        self.middle_layer3 = MaskedConv3d('solid', self.internal_dps, self.internal_dps,
                                          self.kernel_n, self.stride_n, self.pad_n,
                                          bias=self.conv_bias)
        self.middle_layer4 = MaskedConv3d('solid', self.internal_dps, self.internal_dps,
                                          self.kernel_n, self.stride_n, self.pad_n,
                                          bias=self.conv_bias)
        
        self.middle_layer5 = MaskedConv3d('solid', self.internal_dps, self.internal_dps,
                                          self.kernel_n, self.stride_n, self.pad_n,
                                          bias=self.conv_bias)
        self.middle_layer6 = MaskedConv3d('solid', self.internal_dps, self.internal_dps,
                                          self.kernel_n, self.stride_n, self.pad_n,
                                          bias=self.conv_bias)

        self.final_layer = MaskedConv3d('solid', self.internal_dps, self.fm_centers_n,
                                        self.kernel_n, self.stride_n, self.pad_n,
                                        bias=self.conv_bias)

    # def forward(self, input_tensor):
    #     out1 = self.first_layer(input_tensor)
    #     out1 = self.active_func(out1)

    #     out2 = self.middle_layer1(out1)
    #     out2 = self.active_func(out2)
    #     out2 = self.middle_layer2(out2)
    #     out2 = out2+out1

    #     out3 = self.middle_layer3(out2)
    #     out3 = self.active_func(out3)
    #     out3 = self.middle_layer4(out3)
    #     out3 = out3+out2

    #     out4 = self.middle_layer5(out3)
    #     out4 = self.active_func(out4)
    #     out4 = self.middle_layer6(out4)
    #     out4 = out4+out3

    #     out5 = self.final_layer(out4)
    #     out5 = self.active_func(out5)

    #     return out5
    
    def forward(self, input_tensor):
        out1 = self.first_layer(input_tensor)
        out1 = self.active_func(out1)

        out2 = self.middle_layer1(out1)
        out2 = self.active_func(out2)
        out2 = self.middle_layer2(out2)
        out2 = out2+out1

        out3 = self.final_layer(out2)
        out3 = self.active_func(out3)

        return out3


## GMM Quantizer
class GMMQuantizer(nn.Module):
    def __init__(self, num_of_mean, std_list, pi_list, mean_list=None):
        super(GMMQuantizer, self).__init__()
        self.num_of_mean = num_of_mean
        self.sigma = 1.0
        self.hard_sigma = 1e4
        self.div = 1e-3

        if mean_list == None:
            mean_list = np.linspace(-(num_of_mean//2), num_of_mean//2, num_of_mean)
            mean_list = mean_list.astype(np.float32)
            mean = torch.from_numpy(mean_list)
            self.mean = torch.nn.Parameter(mean)
        else:
            mean = torch.from_numpy(np.array(mean_list)).float()
            self.mean = torch.nn.Parameter(mean)

        std = torch.from_numpy(np.array(std_list)).float()
        log_std = torch.log(std)
        self.log_std = torch.nn.Parameter(log_std)
        self.std = None

        pi = torch.from_numpy(np.array(pi_list)).float()
        log_pi = torch.log(pi)
        self.log_pi = torch.nn.Parameter(log_pi)
        self.norm_pi = None


    def forward(self, input_tensor):

        self._get_norm_pi()
        self._get_std()

        dist = self._get_dist_between_mean(input_tensor)
        dist = dist / 2 / (self.std * self.std + self.div)
        phi_soft = self._weighted_softmax(-self.sigma * dist)
        phi_hard = self._weighted_softmax(-self.hard_sigma * dist)
        symbols_hard = torch.argmax(phi_hard, dim=4)
        symbols_hard = symbols_hard.unsqueeze(4)
        shape = symbols_hard.shape
        one_hot = torch.cuda.FloatTensor(shape[0], shape[1], shape[2], shape[3], self.num_of_mean)
        one_hot.zero_()
        one_hot.scatter_(4, symbols_hard, 1)

        softout = torch.sum(phi_soft * self.mean, dim=4)
        hardout = torch.sum(one_hot * self.mean, dim=4)


        mid_tensor_q = softout + (hardout - softout).detach()
        return mid_tensor_q, symbols_hard

    def _get_dist_between_mean(self, input_tensor):
        dist = torch.unsqueeze(input_tensor, 4) - self.mean
        dist = torch.abs(dist)
        dist = torch.mul(dist, dist)
        return dist

    def _weighted_softmax(self, x):
        # x is b x c x h x w x L
        maxes, _ = torch.max(x, dim=4)
        x_exp = torch.exp(x - maxes.unsqueeze(-1)) / torch.sqrt(math.pi * 2 * (self.std * self.std + self.div)) * self.norm_pi 
        # x_exp_sum is b x c x h x w
        x_exp_sum = torch.sum(x_exp, 4)
        return x_exp / x_exp_sum.unsqueeze(4)

    def _get_norm_pi(self):
        min_log_pi = torch.min(self.log_pi)
        mixing_proportions = torch.exp(self.log_pi - min_log_pi)
        self.norm_pi = mixing_proportions / torch.sum(mixing_proportions)
        return

    def _get_std(self):
        self.std = torch.exp(self.log_std)
        return

    def get_prob(self, x):
        """
        x: b x c x h x w
        mean: L
        std: L
        pi: L
        prob: b x c x h x w x L
        out: b x c x h x w x 1
        """
        self._get_norm_pi()
        self._get_std()
        prob = torch.exp(-(x.unsqueeze(4) - self.mean) * (x.unsqueeze(4) - self.mean)/ 2 / (self.std * self.std + self.div)) * self.norm_pi / torch.sqrt(math.pi * 2 * (self.std * self.std + self.div))
        out = torch.sum(prob, 4)
        return out

# Channel attention based learning for channel importance vector 
class ChannelImportance(nn.Module):
    def __init__(self, channel, reduction=1):
        super(ChannelImportance, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y
