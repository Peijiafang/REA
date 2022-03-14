import cv2
import concern.webcv2 as webcv2
import numpy as np
import torch
import pyclipper
from shapely.geometry import Polygon

from concern.config import Configurable, State
from data.processes.make_icdar_data import MakeICDARData


class SegDetectorVisualizer(Configurable):
    vis_num = State(default=4)
    eager_show = State(default=False)

    def __init__(self, **kwargs):
        cmd = kwargs['cmd']
        if 'eager_show' in cmd:
            self.eager_show = cmd['eager_show']

    def visualize(self, batch, output_pair, pred):
        boxes, _ = output_pair
        result_dict = {}
        for i in range(batch['image'].size(0)):
            result_dict.update(
                self.single_visualize(batch, i, boxes[i], pred))
        if self.eager_show:
            webcv2.waitKey()
            return {}
        return result_dict

    def _visualize_heatmap(self, heatmap, canvas=None):
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap[0] * 255).astype(np.uint8)
        if canvas is None:
            pred_image = heatmap
        else:
            pred_image = (heatmap.reshape(
                *heatmap.shape[:2], 1).astype(np.float32) / 255 + 1) / 2 * canvas
            pred_image = pred_image.astype(np.uint8)
        return pred_image


    def single_visualize(self, batch, index, boxes, pred):
        RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        image = batch['image'][index]
        polygons = batch['polygons'][index]
        if isinstance(polygons, torch.Tensor):
            polygons = polygons.cpu().data.numpy()
        ignore_tags = batch['ignore_tags'][index]
        original_shape = batch['shape'][index]
        filename = batch['filename'][index]
        image = (image.cpu().numpy()).transpose(1, 2, 0) * 255 + RGB_MEAN
        pred_canvas = image.copy().astype(np.uint8)
        pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))
        gt = image.copy().astype(np.uint8)
        gt = cv2.resize(gt, (original_shape[1], original_shape[0]))
        ori_seg = np.zeros((original_shape[0], original_shape[1]), dtype=np.float32)
        if isinstance(pred, dict) and 'thresh' in pred:
            thresh = self._visualize_heatmap(pred['thresh'][index])

        if isinstance(pred, dict) and 'thresh_binary' in pred:
            thresh_binary = self._visualize_heatmap(pred['thresh_binary'][index])
            MakeICDARData.polylines(self, thresh_binary, polygons, ignore_tags)

        for i in range(len(polygons)):
            polygon = polygons[i].reshape(-1, 2).astype(np.int32)
            ignore = ignore_tags[i]
            if ignore:
                color = (255, 0, 0)  # depict ignorable polygons in blue
            else:
                color = (0, 0, 255)  # depict polygons in red
            cv2.polylines(gt, [polygon], True, color, 2)

            polygon_shape = Polygon(polygon)
            distance = polygon_shape.area * \
                (1 - np.power(0.4, 2)) / polygon_shape.length
            subject = [tuple(l) for l in polygons[i]]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND,
                            pyclipper.ET_CLOSEDPOLYGON)
            shrinked = padding.Execute(-distance)
            if shrinked == []:
                continue
            shrinked = np.array(shrinked[0]).reshape(-1, 2)

            cv2.fillPoly(ori_seg, [shrinked.astype(np.int32)], 255)

        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)
            # if isinstance(pred, dict) and 'thresh_binary' in pred:
            #     cv2.polylines(thresh_binary, [box], True, (0, 255, 0), 1) 
        
        if isinstance(pred, dict):
            if 'zooming_preds' in pred:
                prob = (pred['zooming_preds']['select'][0].cpu().numpy()).transpose(1, 2, 0) * 255
            else:
                prob = (pred['binary'][0].cpu().numpy()).transpose(1, 2, 0) * 255
        else:
            prob = (pred[0].cpu().numpy()).transpose(1, 2, 0) * 255
            
        # prob = cv2.resize(prob, (original_shape[1], original_shape[0]))
        # prob = np.tile(np.expand_dims(prob, 2), (1, 1, 3))

        prob = cv2.resize(prob, (original_shape[1], original_shape[0]))
        prob = np.tile(np.expand_dims(prob, 2), (1, 1, 3))

        # thresh_binary = (pred['thresh_binary'][0].cpu().numpy()).transpose(1, 2, 0) * 255
        # thresh_binary = cv2.resize(thresh_binary, (original_shape[1], original_shape[0]))
        # thresh_binary = np.tile(np.expand_dims(thresh_binary, 2), (1, 1, 3))

        ones = np.ones([1, original_shape[1], 3]) * 255
        mix = np.concatenate([pred_canvas, ones, prob], axis=0)
        # mix = np.concatenate([pred_canvas, ones, prob, ones, thresh_binary], axis=0)

        if self.eager_show:
            webcv2.imshow(filename + ' output', cv2.resize(pred_canvas, (1024, 1024)))
            if isinstance(pred, dict) and 'thresh' in pred:
                webcv2.imshow(filename + ' thresh', cv2.resize(thresh, (1024, 1024)))
                webcv2.imshow(filename + ' pred', cv2.resize(pred_canvas, (1024, 1024)))
            if isinstance(pred, dict) and 'thresh_binary' in pred:
                webcv2.imshow(filename + ' thresh_binary', cv2.resize(thresh_binary, (1024, 1024)))
            return {}
        else:
            
            if isinstance(pred, dict) and 'thresh' in pred:
                return {
                    filename + '_output': pred_canvas,
                    filename + '_gt': gt,
                    # filename + '_prob': prob,
                    # filename + '_mix' : mix,
                    filename + '_ori_seg' : ori_seg,
                    # filename + '_thresh': thresh,
                    # filename + '_pred': thresh_binary
                }
            else:
                return {
                    filename + '_output': pred_canvas,
                    filename + '_mix' : mix,
                    # filename + '_prob' : prob,
                    # filename + '_pred': thresh_binary
            }

    def demo_visualize(self, image_path, output):
        boxes, _ = output
        boxes = boxes[0]
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_shape = original_image.shape
        pred_canvas = original_image.copy().astype(np.uint8)
        pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 0, 255), 2)

        return pred_canvas

