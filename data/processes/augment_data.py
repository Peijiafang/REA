import imgaug
import numpy as np

from concern.config import State
from .data_process import DataProcess
from data.augmenter import AugmenterBuilder
import cv2
import math


class AugmentData(DataProcess):
    augmenter_args = State(autoload=False)

    def __init__(self, **kwargs):
        self.augmenter_args = kwargs.get('augmenter_args')
        self.keep_ratio = kwargs.get('keep_ratio')
        self.only_resize = kwargs.get('only_resize')
        self.super_resolution = kwargs.get('super_resolution')
        self.blur_gt = kwargs.get('use_blur')
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def blured_image(self, image):
        alpha = np.random.rand(2)
        h, w, _ = image.shape
        # 0 : blur
        # 1 : down-up sample
        if alpha[0] >= 0.5 and alpha[1] >= 0.5:
            image = cv2.blur(image, (7, 7))
            image = cv2.resize(image, (w // 4, h // 4))
            image = cv2.resize(image, (w, h))
        elif alpha[0] >= 0.5 and alpha[1] < 0.5:
            image = cv2.blur(image, (7, 7))
        elif alpha[0] < 0.5 and alpha[1] >= 0.5:
            image = cv2.resize(image, (w // 4, h // 4))
            image = cv2.resize(image, (w, h))
        return image

    def process(self, data):
        image = data['image']
        aug = None
        shape = image.shape
        data['super_resolution'] = self.super_resolution

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['image'] = self.resize_image(image)
            else:
                if self.super_resolution:
                    image_blur = self.blured_image(image)
                    data['image_blur'] = aug.augment_image(image_blur)
                data['image'] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        if not self.only_resize:
            data['is_training'] = True
        else:
            data['is_training'] = False
        return data


class AugmentDetectionData(AugmentData):
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': False if self.blur_gt else line['text'] == '###',
                'text': line['text'],
            })
        data['polys'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
