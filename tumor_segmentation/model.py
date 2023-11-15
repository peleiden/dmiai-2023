import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from . import device, TrainConfig
from tumor_segmentation.dice import BinaryDiceLoss


class TumorBoi(nn.Module):

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        self.processor = AutoImageProcessor.from_pretrained(self.config.pretrain_path)
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.config.pretrain_path,
            config = self.config.config,
            ignore_mismatched_sizes = True,
        )
        self._set_dropouts(self.config.dropout)

    def forward(self, images: list[np.ndarray], labels: list[np.ndarray]):
        inputs = self.processor.preprocess(images, labels, return_tensors="pt")
        images = inputs['pixel_values'].to(device)
        mask_labels = [x.to(device) for x in inputs['mask_labels']]
        class_labels = [x.to(device) for x in inputs['class_labels']]
        out = self.mask2former(
            pixel_values = images,
            mask_labels  = mask_labels,
            class_labels = class_labels,
        )
        return out

    def out_to_seg(self, out, img: np.ndarray) -> np.ndarray:
        seg = self.processor.post_process_semantic_segmentation(out)[0].cpu().numpy().astype(np.uint8)
        seg = cv2.resize(seg, (img.shape[1], img.shape[0])).astype(bool)
        return seg

    def out_to_segs(self, out, original_shapes) -> list[np.ndarray]:
        segs = self.processor.post_process_semantic_segmentation(out)
        segs = [seg.cpu().numpy().astype(np.uint8) for seg in segs]
        return [np.array(cv2.resize(seg, original_shape[::-1]).astype(bool)) for original_shape, seg in zip(original_shapes, segs)]

    def _set_dropouts(self, p: float):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = p
