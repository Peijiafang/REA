from collections import OrderedDict

import torch
import torch.nn as nn
BatchNorm2d = nn.BatchNorm2d
import numpy as np
class SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.

        # binary = self.binarize(fuse)

        # if self.training:
        #     result = OrderedDict(binary=binary)
        # else:
        #     return binary
        # if self.adaptive and self.training:
        #     if self.serial:
        #         fuse = torch.cat(
        #                 (fuse, nn.functional.interpolate(
        #                     binary, fuse.shape[2:])), 1)
        #     thresh = self.thresh(fuse)
        #     thresh_binary = self.step_function(binary, thresh)
        #     result.update(thresh=thresh, thresh_binary=thresh_binary)
        # return result

        binary = self.binarize(fuse)
        result = OrderedDict(binary=binary)
        if self.training and not self.adaptive:
            return result
        if self.adaptive:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BiFPN_1layer(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(BiFPN_1layer, self).__init__()
        self.k = k
        self.serial = serial

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.BiFPN = BiFPN_cnn(inner_channels, bias, weighted=True)
        self.BiFPN.apply(self.weights_init)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        b2, b3, b4, b5 = self.BiFPN(in2, in3, in4, in5)

        p2 = self.out2(b2)
        p3 = self.out3(b3)
        p4 = self.out4(b4)
        p5 = self.out5(b5)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        result = OrderedDict(binary=binary)
        if self.training and not self.adaptive:
            return result
        if self.adaptive:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BiFPN_1layer_CEMadd(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False, 
                 normalize=True,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(BiFPN_1layer_CEMadd, self).__init__()
        self.k = k
        self.serial = serial

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.BiFPN = BiFPN_cnn(inner_channels, bias, weighted=True)
        self.BiFPN.apply(self.weights_init)

        self.CEM = CEM(64, normalize=normalize)
        self.CEM.apply(self.weights_init)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        b2, b3, b4, b5 = self.BiFPN(in2, in3, in4, in5)

        p2 = self.out2(b2)
        p3 = self.out3(b3)
        p4 = self.out4(b4)
        p5 = self.out5(b5)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.CEM(fuse) + fuse
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BiFPN_1layer_SR(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        super(BiFPN_1layer_SR, self).__init__()
        self.k = k
        self.serial = serial

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.BiFPN = BiFPN_cnn(inner_channels, bias, weighted=True)
        self.BiFPN.apply(self.weights_init)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)
        # why set reduction?
        # 512 => 32
        # 64 => 32 ?
        self.sr = SR(n_RG=2, n_RCAB=2, inner_channels=inner_channels // 4, reduction=2, bias=bias)
        self.sr.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        b2, b3, b4, b5 = self.BiFPN(in2, in3, in4, in5)

        p2 = self.out2(b2)
        p3 = self.out3(b3)
        p4 = self.out4(b4)
        p5 = self.out5(b5)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)

        sr = self.sr(p5)
        result.update(sr=sr)

        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BiFPN_1layer_bnrelu(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(BiFPN_1layer_bnrelu, self).__init__()
        self.k = k
        self.serial = serial

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.BiFPN = BiFPN_cnn_bnrelu(inner_channels, bias, weighted=True)
        self.BiFPN.apply(self.weights_init)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        b2, b3, b4, b5 = self.BiFPN(in2, in3, in4, in5)

        p2 = self.out2(b2)
        p3 = self.out3(b3)
        p4 = self.out4(b4)
        p5 = self.out5(b5)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class SegDetector_BiFPN(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False, weight_bifpn=True,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector_BiFPN, self).__init__()
        self.k = k
        self.serial = serial
        self.outer_channels = inner_channels // 4
        self.weight_bifpn = weight_bifpn
        if weight_bifpn:

            self.pt4_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.pt3_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out2_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out3_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out4_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out5_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)

            self.register_parameter("pt4_w", self.pt4_w)
            self.register_parameter("pt3_w", self.pt3_w)
            self.register_parameter("out2_w", self.out2_w)
            self.register_parameter("out3_w", self.out3_w)
            self.register_parameter("out4_w", self.out4_w)
            self.register_parameter("out5_w", self.out5_w)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.pt3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))
        self.pt4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))

        self.out5_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))
        self.out4_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))
        self.out3_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))
        self.out2_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))

        self.p5 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.p4 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.p3 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.p2 = nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias)

        # DB branch does not change
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 3, padding=1, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.outer_channels, self.outer_channels, 2, 2),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.outer_channels, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.pt4.apply(self.weights_init)
        self.pt3.apply(self.weights_init)
        self.out5_d.apply(self.weights_init)
        self.out4_d.apply(self.weights_init)
        self.out3_d.apply(self.weights_init)
        self.out2_d.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, self.outer_channels, 3, padding=1, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            self._init_upsample(self.outer_channels, self.outer_channels, smooth=smooth, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            self._init_upsample(self.outer_channels, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        if self.weight_bifpn:
            pt4_w = torch.relu(self.pt4_w)
            pt4 = self.pt4((pt4_w[0] * in4 + pt4_w[1] * self.up5(in5)) / torch.sum(pt4_w))
            pt3_w = torch.relu(self.pt3_w)
            pt3 = self.pt3((pt3_w[0] * in3 + pt3_w[0] * self.up4(pt4)) / torch.sum(pt3_w))

            out2_w = torch.relu(self.out2_w)
            out2_d = self.out2_d((out2_w[0] * in2 + out2_w[0] * self.up3(pt3)) / torch.sum(out2_w))
            out3_w = torch.relu(self.out3_w)
            out3_d = self.out3_d((out3_w[0] * in3 + out3_w[1] * pt3 + out3_w[2] * nn.functional.interpolate(out2_d, scale_factor=0.5)) / torch.sum(out3_w))
            out4_w = torch.relu(self.out4_w)
            out4_d = self.out4_d((out4_w[0] * in4 + out4_w[1] * pt4 + out4_w[2] * nn.functional.interpolate(out3_d, scale_factor=0.5)) / torch.sum(out4_w))
            out5_w = torch.relu(self.out5_w)
            out5_d = self.out5_d((out5_w[0] * in5 + out5_w[1] * nn.functional.interpolate(out4_d, scale_factor=0.5)) / torch.sum(out5_w))

        else:
            pt4 = self.pt4(in4 + self.up5(in5))
            pt3 = self.pt3(in3 + self.up4(pt4))

            out2_d = self.out2_d(in2 + self.up3(pt3))
            out3_d = self.out3_d(in3 + pt3 + nn.functional.interpolate(out2_d, scale_factor=0.5))
            out4_d = self.out4_d(in4 + pt4 + nn.functional.interpolate(out3_d, scale_factor=0.5))
            out5_d = self.out5_d(in5 + nn.functional.interpolate(out4_d, scale_factor=0.5))

        p5 = self.p5(out5_d)
        p4 = self.p4(out4_d)
        p3 = self.p3(out3_d)
        p2 = self.p2(out2_d)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BiFPN_3layer_bnrelu(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False, 
                 weight_bifpn=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(BiFPN_3layer_bnrelu, self).__init__()
        self.k = k
        self.serial = serial
        self.outer_channels = inner_channels // 4

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.BiFPN1 = BiFPN_bnrelu(inner_channels, bias, weight_bifpn)
        self.BiFPN2 = BiFPN_bnrelu(inner_channels, bias, weight_bifpn)
        self.BiFPN3 = BiFPN_bnrelu(inner_channels, bias, weight_bifpn)
        self.BiFPN1.apply(self.weights_init)
        self.BiFPN2.apply(self.weights_init)
        self.BiFPN3.apply(self.weights_init)

        self.p5 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.p4 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.p3 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.p2 = nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias)

        # DB branch does not change
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 3, padding=1, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.outer_channels, self.outer_channels, 2, 2),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.outer_channels, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.p5.apply(self.weights_init)
        self.p4.apply(self.weights_init)
        self.p3.apply(self.weights_init)
        self.p2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, self.outer_channels, 3, padding=1, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            self._init_upsample(self.outer_channels, self.outer_channels, smooth=smooth, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            self._init_upsample(self.outer_channels, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        b1_1, b1_2, b1_3, b1_4 = self.BiFPN1(in2, in3, in4, in5)
        b2_1, b2_2, b2_3, b2_4 = self.BiFPN2(b1_1, b1_2, b1_3, b1_4)
        out2_d, out3_d, out4_d, out5_d = self.BiFPN3(b2_1, b2_2, b2_3, b2_4)

        p5 = self.p5(out5_d)
        p4 = self.p4(out4_d)
        p3 = self.p3(out3_d)
        p2 = self.p2(out2_d)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BiFPN_CEM(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False, 
                 weight_bifpn=True, cem_normalize=True,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(BiFPN_CEM, self).__init__()
        self.k = k
        self.serial = serial
        self.outer_channels = inner_channels // 4

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.BiFPN1 = BiFPN(inner_channels, bias, weight_bifpn)
        self.BiFPN2 = BiFPN(inner_channels, bias, weight_bifpn)
        self.BiFPN3 = BiFPN(inner_channels, bias, weight_bifpn)
        self.BiFPN1.apply(self.weights_init)
        self.BiFPN2.apply(self.weights_init)
        self.BiFPN3.apply(self.weights_init)

        self.p5 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.p4 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.p3 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.p2 = nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias)

        self.CEM = CEM(64, normalize=cem_normalize)
        self.CEM.apply(self.weights_init)

        # DB branch does not change
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 3, padding=1, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.outer_channels, self.outer_channels, 2, 2),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.outer_channels, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.p5.apply(self.weights_init)
        self.p4.apply(self.weights_init)
        self.p3.apply(self.weights_init)
        self.p2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        b1_1, b1_2, b1_3, b1_4 = self.BiFPN1(in2, in3, in4, in5)
        b2_1, b2_2, b2_3, b2_4 = self.BiFPN2(b1_1, b1_2, b1_3, b1_4)
        out2_d, out3_d, out4_d, out5_d = self.BiFPN3(b2_1, b2_2, b2_3, b2_4)

        p5 = self.p5(out5_d)
        p4 = self.p4(out4_d)
        p3 = self.p3(out3_d)
        p2 = self.p2(out2_d)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.CEM(fuse)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BiFPN_SGE(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False, weight_bifpn=True,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(BiFPN_SGE, self).__init__()
        self.k = k
        self.serial = serial
        self.outer_channels = inner_channels // 4

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.BiFPN = BiFPN(inner_channels, bias, weight_bifpn)
        self.BiFPN.apply(self.weights_init)

        self.p5 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.p4 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.p3 = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.p2 = nn.Conv2d(inner_channels, self.outer_channels, 1, padding=0, bias=bias)

        self.SGE = SGE(64)
        self.SGE.apply(self.weights_init)

        # DB branch does not change
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, self.outer_channels, 3, padding=1, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.outer_channels, self.outer_channels, 2, 2),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.outer_channels, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, self.outer_channels, 3, padding=1, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            self._init_upsample(self.outer_channels, self.outer_channels, smooth=smooth, bias=bias),
            BatchNorm2d(self.outer_channels),
            nn.ReLU(inplace=True),
            self._init_upsample(self.outer_channels, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out2_d, out3_d, out4_d, out5_d = self.BiFPN(in2, in3, in4, in5)

        p5 = self.p5(out5_d)
        p4 = self.p4(out4_d)
        p3 = self.p3(out3_d)
        p2 = self.p2(out2_d)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.SGE(fuse)
        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            binary, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


# module

class BiFPN(nn.Module):
    def __init__(self, inner_channels, bias, weighted):
        """BiFPN with depthwise convolutional layer

        Arguments:
            inner_channels {int} -- [description]
            bias {bool} -- whether use bias
            weighted {bool} -- whether use weighte for addition
        """
        super(BiFPN, self).__init__()
        self.outer_channels = inner_channels // 4
        self.weighted = weighted

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        if self.weighted:
            self.pt4_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.pt3_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out2_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out3_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out4_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out5_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)

            self.register_parameter("pt4_w", self.pt4_w)
            self.register_parameter("pt3_w", self.pt3_w)
            self.register_parameter("out2_w", self.out2_w)
            self.register_parameter("out3_w", self.out3_w)
            self.register_parameter("out4_w", self.out4_w)
            self.register_parameter("out5_w", self.out5_w)

        self.pt3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))
        self.pt4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))

        self.out5_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))
        self.out4_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))
        self.out3_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))
        self.out2_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias))

    def forward(self, in2, in3, in4, in5):
        if self.weighted:
            pt4_w = torch.relu(self.pt4_w)
            pt4 = self.pt4((pt4_w[0] * in4 + pt4_w[1] * self.up5(in5)) / torch.sum(pt4_w))
            pt3_w = torch.relu(self.pt3_w)
            pt3 = self.pt3((pt3_w[0] * in3 + pt3_w[0] * self.up4(pt4)) / torch.sum(pt3_w))

            out2_w = torch.relu(self.out2_w)
            out2_d = self.out2_d((out2_w[0] * in2 + out2_w[0] * self.up3(pt3)) / torch.sum(out2_w))
            out3_w = torch.relu(self.out3_w)
            out3_d = self.out3_d((out3_w[0] * in3 + out3_w[1] * pt3 + out3_w[2] * nn.functional.interpolate(out2_d, scale_factor=0.5)) / torch.sum(out3_w))
            out4_w = torch.relu(self.out4_w)
            out4_d = self.out4_d((out4_w[0] * in4 + out4_w[1] * pt4 + out4_w[2] * nn.functional.interpolate(out3_d, scale_factor=0.5)) / torch.sum(out4_w))
            out5_w = torch.relu(self.out5_w)
            out5_d = self.out5_d((out5_w[0] * in5 + out5_w[1] * nn.functional.interpolate(out4_d, scale_factor=0.5)) / torch.sum(out5_w))

        else:
            pt4 = self.pt4(in4 + self.up5(in5))
            pt3 = self.pt3(in3 + self.up4(pt4))

            out2_d = self.out2_d(in2 + self.up3(pt3))
            out3_d = self.out3_d(in3 + pt3 + nn.functional.interpolate(out2_d, scale_factor=0.5))
            out4_d = self.out4_d(in4 + pt4 + nn.functional.interpolate(out3_d, scale_factor=0.5))
            out5_d = self.out5_d(in5 + nn.functional.interpolate(out4_d, scale_factor=0.5))

        return out2_d, out3_d, out4_d, out5_d


class BiFPN_cnn(nn.Module):
    def __init__(self, inner_channels, bias, weighted):
        """BiFPN with traditional convolutional layer

        Arguments:
            inner_channels {int} -- [description]
            bias {bool} -- whether use bias
            weighted {bool} -- whether use weighte for addition
        """
        super(BiFPN_cnn, self).__init__()
        self.outer_channels = inner_channels // 4
        self.weighted = weighted

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        if self.weighted:
            self.pt4_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.pt3_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out2_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out3_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out4_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out5_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)

            self.register_parameter("pt4_w", self.pt4_w)
            self.register_parameter("pt3_w", self.pt3_w)
            self.register_parameter("out2_w", self.out2_w)
            self.register_parameter("out3_w", self.out3_w)
            self.register_parameter("out4_w", self.out4_w)
            self.register_parameter("out5_w", self.out5_w)

        self.pt3 = nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias)
        self.pt4 = nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias)

        self.out5_d = nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias)
        self.out4_d = nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias)
        self.out3_d = nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias)
        self.out2_d = nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias)

    def forward(self, in2, in3, in4, in5):
        if self.weighted:
            pt4_w = torch.relu(self.pt4_w)
            pt4 = self.pt4((pt4_w[0] * in4 + pt4_w[1] * self.up5(in5)) / torch.sum(pt4_w))
            pt3_w = torch.relu(self.pt3_w)
            pt3 = self.pt3((pt3_w[0] * in3 + pt3_w[0] * self.up4(pt4)) / torch.sum(pt3_w))

            out2_w = torch.relu(self.out2_w)
            out2_d = self.out2_d((out2_w[0] * in2 + out2_w[0] * self.up3(pt3)) / torch.sum(out2_w))
            out3_w = torch.relu(self.out3_w)
            out3_d = self.out3_d((out3_w[0] * in3 + out3_w[1] * pt3 + out3_w[2] * nn.functional.interpolate(out2_d, scale_factor=0.5)) / torch.sum(out3_w))
            out4_w = torch.relu(self.out4_w)
            out4_d = self.out4_d((out4_w[0] * in4 + out4_w[1] * pt4 + out4_w[2] * nn.functional.interpolate(out3_d, scale_factor=0.5)) / torch.sum(out4_w))
            out5_w = torch.relu(self.out5_w)
            out5_d = self.out5_d((out5_w[0] * in5 + out5_w[1] * nn.functional.interpolate(out4_d, scale_factor=0.5)) / torch.sum(out5_w))

        else:
            pt4 = self.pt4(in4 + self.up5(in5))
            pt3 = self.pt3(in3 + self.up4(pt4))

            out2_d = self.out2_d(in2 + self.up3(pt3))
            out3_d = self.out3_d(in3 + pt3 + nn.functional.interpolate(out2_d, scale_factor=0.5))
            out4_d = self.out4_d(in4 + pt4 + nn.functional.interpolate(out3_d, scale_factor=0.5))
            out5_d = self.out5_d(in5 + nn.functional.interpolate(out4_d, scale_factor=0.5))

        return out2_d, out3_d, out4_d, out5_d


class BiFPN_cnn_bnrelu(nn.Module):
    def __init__(self, inner_channels, bias, weighted):
        """BiFPN with traditional convolutional layer + bn +relu

        Arguments:
            inner_channels {int} -- [description]
            bias {bool} -- whether use bias
            weighted {bool} -- whether use weighte for addition
        """
        super(BiFPN_cnn_bnrelu, self).__init__()
        self.outer_channels = inner_channels // 4
        self.weighted = weighted

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        if self.weighted:
            self.pt4_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.pt3_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out2_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out3_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out4_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out5_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)

            self.register_parameter("pt4_w", self.pt4_w)
            self.register_parameter("pt3_w", self.pt3_w)
            self.register_parameter("out2_w", self.out2_w)
            self.register_parameter("out3_w", self.out3_w)
            self.register_parameter("out4_w", self.out4_w)
            self.register_parameter("out5_w", self.out5_w)

        self.pt3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        self.pt4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))

        self.out5_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        self.out4_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        self.out3_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        self.out2_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))

    def forward(self, in2, in3, in4, in5):
        if self.weighted:
            pt4_w = torch.relu(self.pt4_w)
            pt4 = self.pt4((pt4_w[0] * in4 + pt4_w[1] * self.up5(in5)) / torch.sum(pt4_w))
            pt3_w = torch.relu(self.pt3_w)
            pt3 = self.pt3((pt3_w[0] * in3 + pt3_w[0] * self.up4(pt4)) / torch.sum(pt3_w))

            out2_w = torch.relu(self.out2_w)
            out2_d = self.out2_d((out2_w[0] * in2 + out2_w[0] * self.up3(pt3)) / torch.sum(out2_w))
            out3_w = torch.relu(self.out3_w)
            out3_d = self.out3_d((out3_w[0] * in3 + out3_w[1] * pt3 + out3_w[2] * nn.functional.interpolate(out2_d, scale_factor=0.5)) / torch.sum(out3_w))
            out4_w = torch.relu(self.out4_w)
            out4_d = self.out4_d((out4_w[0] * in4 + out4_w[1] * pt4 + out4_w[2] * nn.functional.interpolate(out3_d, scale_factor=0.5)) / torch.sum(out4_w))
            out5_w = torch.relu(self.out5_w)
            out5_d = self.out5_d((out5_w[0] * in5 + out5_w[1] * nn.functional.interpolate(out4_d, scale_factor=0.5)) / torch.sum(out5_w))

        else:
            pt4 = self.pt4(in4 + self.up5(in5))
            pt3 = self.pt3(in3 + self.up4(pt4))

            out2_d = self.out2_d(in2 + self.up3(pt3))
            out3_d = self.out3_d(in3 + pt3 + nn.functional.interpolate(out2_d, scale_factor=0.5))
            out4_d = self.out4_d(in4 + pt4 + nn.functional.interpolate(out3_d, scale_factor=0.5))
            out5_d = self.out5_d(in5 + nn.functional.interpolate(out4_d, scale_factor=0.5))

        return out2_d, out3_d, out4_d, out5_d


class BiFPN_bnrelu(nn.Module):
    def __init__(self, inner_channels, bias, weighted):
        """BiFPN with depthwise convolutional layer + bn +relu

        Arguments:
            inner_channels {int} -- [description]
            bias {bool} -- whether use bias
            weighted {bool} -- whether use weighte for addition
        """
        super(BiFPN_bnrelu, self).__init__()
        self.outer_channels = inner_channels // 4
        self.weighted = weighted

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        if self.weighted:
            self.pt4_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.pt3_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out2_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)
            self.out3_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out4_w = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
            self.out5_w = nn.Parameter(torch.Tensor([1, 1]), requires_grad=True)

            self.register_parameter("pt4_w", self.pt4_w)
            self.register_parameter("pt3_w", self.pt3_w)
            self.register_parameter("out2_w", self.out2_w)
            self.register_parameter("out3_w", self.out3_w)
            self.register_parameter("out4_w", self.out4_w)
            self.register_parameter("out5_w", self.out5_w)

        self.pt3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        self.pt4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))

        self.out5_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        self.out4_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        self.out3_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))
        self.out2_d = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias, groups=inner_channels),
            nn.Conv2d(inner_channels, inner_channels, 1, padding=0, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True))

    def forward(self, in2, in3, in4, in5):
        if self.weighted:
            pt4_w = torch.relu(self.pt4_w)
            pt4 = self.pt4((pt4_w[0] * in4 + pt4_w[1] * self.up5(in5)) / torch.sum(pt4_w))
            pt3_w = torch.relu(self.pt3_w)
            pt3 = self.pt3((pt3_w[0] * in3 + pt3_w[0] * self.up4(pt4)) / torch.sum(pt3_w))

            out2_w = torch.relu(self.out2_w)
            out2_d = self.out2_d((out2_w[0] * in2 + out2_w[0] * self.up3(pt3)) / torch.sum(out2_w))
            out3_w = torch.relu(self.out3_w)
            out3_d = self.out3_d((out3_w[0] * in3 + out3_w[1] * pt3 + out3_w[2] * nn.functional.interpolate(out2_d, scale_factor=0.5)) / torch.sum(out3_w))
            out4_w = torch.relu(self.out4_w)
            out4_d = self.out4_d((out4_w[0] * in4 + out4_w[1] * pt4 + out4_w[2] * nn.functional.interpolate(out3_d, scale_factor=0.5)) / torch.sum(out4_w))
            out5_w = torch.relu(self.out5_w)
            out5_d = self.out5_d((out5_w[0] * in5 + out5_w[1] * nn.functional.interpolate(out4_d, scale_factor=0.5)) / torch.sum(out5_w))

        else:
            pt4 = self.pt4(in4 + self.up5(in5))
            pt3 = self.pt3(in3 + self.up4(pt4))

            out2_d = self.out2_d(in2 + self.up3(pt3))
            out3_d = self.out3_d(in3 + pt3 + nn.functional.interpolate(out2_d, scale_factor=0.5))
            out4_d = self.out4_d(in4 + pt4 + nn.functional.interpolate(out3_d, scale_factor=0.5))
            out5_d = self.out5_d(in5 + nn.functional.interpolate(out4_d, scale_factor=0.5))

        return out2_d, out3_d, out4_d, out5_d


class SGE(nn.Module):
    """SpatialGroupEnhance"""
    def __init__(self, groups):
        super(SGE, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)
        self.sig = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.avg_pool(x)  # bg a h w * bg h w
        xn = xn.sum(dim=1, keepdim=True)  # bg 1 h w
        t = xn.view(b * self.groups, -1)  # bg hw
        t = t - t.mean(dim=1, keepdim=True)  # bg hw
        std = t.std(dim=1, keepdim=True) + 1e-5  # bg 1
        t = t / std  # bg hw
        t = t.view(b, self.groups, h, w)  # b g h w
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class CEM(nn.Module):
    """CapsuleEnhanceModule"""
    def __init__(self, groups, normalize=False):
        super(CEM, self).__init__()

        self.groups = groups
        self.normalize = normalize
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)
        self.sig = nn.Sigmoid()

    def __call__(self, x):
        b, c, h, w = x.size()
        x = x.view(b, self.groups, -1, h, w)

        if self.normalize:
            xa = self._squash_factor(x, keepdim=True)  # b g 1 h w
            xn = x * xa  # b g a h w
            xn = xn.sum(dim=2, keepdim=True)  # b g 1 h w
            t = xn.view(b, self.groups, -1)  # b g hw
            t = t - t.mean(dim=2, keepdim=True)  # b g hw
            std = t.std(dim=2, keepdim=True) + 1e-5  # b g 1
            t = t / std  # b g hw
            t = t.view(b, self.groups, h, w)
            t = self.weight * t + self.bias
            t = self.sig(t)
        else:
            xa = self._squash_factor(x, keepdim=False)  # b g h w
            t = self.weight * xa + self.bias  # b g h w
            t = torch.clamp(t, min=0.0, max=1.0)

        t = t.view(b, self.groups, 1, h, w)
        x = x * t
        x = x.view(b, c, h, w)
        return x

    def _squash_factor(self, vector, keepdim=True):
        # b g a h w
        vec_squared = vector.pow(2).sum(dim=2, keepdim=keepdim)
        scalar_factor = vec_squared / (1 + vec_squared)
        return scalar_factor


# SR
class RCAB(nn.Module):
    def __init__(self, inner_channels, reduction, bias):
        """Residual Channel Attention Block"""
        super(RCAB, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels // reduction, inner_channels, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.pre(x)
        a = self.CA(self.avg_pool(res))
        res *= a
        res += x
        return res


class RG(nn.Module):
    def __init__(self, n_RCAB, inner_channels, reduction, bias):
        """Residual Group"""
        super(RG, self).__init__()
        modules_body = [RCAB(inner_channels, reduction, bias=bias) for _ in range(n_RCAB)]
        modules_body.append(nn.Conv2d(inner_channels, inner_channels, 3, padding=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class SR(nn.Module):
    def __init__(self, n_RG, n_RCAB, inner_channels, reduction, bias):
        "super-resolution"
        super(SR, self).__init__()
        modules_body = [RG(n_RCAB, inner_channels, reduction, bias) for _ in range(n_RG)]
        modules_body.append(nn.Conv2d(inner_channels, inner_channels, 3, padding=1))

        # define tail module
        modules_tail = [
            nn.Conv2d(inner_channels, 4 * inner_channels, 3, padding=1, bias=bias),
            nn.PixelShuffle(2),
            nn.Conv2d(inner_channels, 4 * inner_channels, 3, padding=1, bias=bias),
            nn.PixelShuffle(2),
            nn.Conv2d(inner_channels, 3, 3, padding=1, bias=bias)]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

