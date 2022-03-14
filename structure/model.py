import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
import cv2

import backbones
import decoders


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)
        self.with_zooming = args['with_zooming']
        if self.with_zooming:
            self.zooming = Zooming()
            self.select = Select()
            self.select = parallelize(self.select, distributed, local_rank)

    @staticmethod
    def model_name(args):
        return os.path.join(args['dataset'], args['backbone'], args['decoder'], args['loss_class'])
        # return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, training=self.training)

        if self.with_zooming:
            output = self.zooming(data.detach(), pred['binary'].detach(), self.device)  # tlbr
            zooming_image = output['zooming_image']
            coordinates = output['coordinates']
            zooming_preds = self.model(zooming_image.detach(), training=self.training)
            # cv2.imwrite('demo_results/crop_ori_bi.jpg',zooming_preds['binary'][0].cpu().numpy().transpose(1,2,0) * 255)

            for key, value in list(zooming_preds.items()):
                batch_size, channel, height, width = value.size()
                local_imgs = torch.zeros([batch_size, channel, height, width]).to(self.device)
                for i, coordinate in enumerate(coordinates):
                    t, l, b, r = coordinate
                    if key == 'thresh':
                        local_imgs[i, :, t:b + 1, l:r + 1] = F.interpolate(value[i].unsqueeze(0), size=(b - t + 1, r - l + 1), mode='bilinear')
                    else:
                        local_imgs[i, :, t:b + 1, l:r + 1] = F.interpolate(value[i].unsqueeze(0), size=(b - t + 1, r - l + 1), mode='nearest')
                zooming_preds[key] = local_imgs
                
            select_output = self.select(pred, zooming_preds)
            zooming_preds.update(zooming_image=zooming_image, select=select_output, coordinates=coordinates)
            pred.update(zooming_preds=zooming_preds)

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred


class SR_SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SR_SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join(args['dataset'], args['backbone'], args['decoder'], args['loss_class'])
        # return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            if training:
                data = batch['image_blur'].to(self.device)
            else:
                data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, training=self.training)

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred


class Zooming(nn.Module):
    def __init__(self, threshold=0.05):
        super(Zooming, self).__init__()
        self.threshold = threshold

    def forward(self, ori, segs, DEVICE='cuda'):
        b, c, h, w = segs.size()

        local_imgs = torch.zeros([b, 3, h, w]).to(DEVICE)
        coordinates = []
        for i, seg in enumerate(segs):
            seg = seg.detach().cpu().numpy().transpose(1, 2, 0)
            seg_np_thresh = (seg > self.threshold)
            
            objs = ndimage.find_objects(seg_np_thresh)  #(top, bot) (left, right)
            if len(objs) == 1:
                objs = objs[0]
                left = int(objs[1].start * 0.9) if int(objs[1].start * 0.9) >= 0 else 0
                top = int(objs[0].start * 0.9) if int(objs[0].start * 0.9) >= 0 else 0
                right = int(objs[1].stop * 1.1) if int(objs[1].stop * 1.1) < w else w - 1
                bot = int(objs[0].stop * 1.1) if int(objs[0].stop * 1.1) < h else h - 1
                coordinates.append([top, left, bot, right])
                
                local_imgs[i:i + 1] = F.interpolate(ori[i:i + 1, :, top:(bot + 1), left:(right + 1)], size=(h, w), mode='bilinear', align_corners=True)
            else:
                coordinates.append([0, 0, h - 1, w - 1])
                local_imgs[i:i + 1] = ori[i:i + 1, :, :, :]
        output = dict(zooming_image=local_imgs, coordinates=coordinates)
        return output


class Select(nn.Module):
    def __init__(self):
        super(Select, self).__init__()
        self.select_binary = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=False),
            nn.Sigmoid())

        self.select_binary2 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=False),
            nn.Sigmoid())


        # self.select_binary.apply(self.weights_init)
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    
    def forward(self, pred, zooming_pred):
        # fuse = torch.cat((pred['binary'], zooming_pred['binary']), axis=1)
        if 'thresh_binary' in pred and 'thresh_binary' in zooming_pred:
            fuse = torch.cat((pred['binary'], pred['thresh_binary'], zooming_pred['binary'], zooming_pred['thresh_binary']), axis=1)
            select_binary = self.select_binary(fuse)
        else:
            fuse = torch.cat((pred['binary'], zooming_pred['binary']), axis=1)
            select_binary = self.select_binary2(fuse)
        # print(torch.sum(fuse,axis=1))
        # fuse = pred['thresh_binary'] + zooming_pred['thresh_binary']
        return select_binary
