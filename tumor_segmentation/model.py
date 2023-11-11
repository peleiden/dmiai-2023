import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from . import device, TrainConfig


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

    def forward(self, images: np.ndarray, labels: np.ndarray) -> torch.Tensor:
        inputs = self.processor.preprocess(list(images), list(labels), return_tensors="pt")
        images = inputs['pixel_values'].to(device)
        mask_labels = [x.to(device) for x in inputs['mask_labels']]
        class_labels = [x.to(device) for x in inputs['class_labels']]
        out = self.mask2former(
            pixel_values = images,
            mask_labels  = mask_labels,
            class_labels = class_labels,
        )
        return out
