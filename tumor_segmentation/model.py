import cv2
import numpy as np
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from . import device, TrainConfig


def pad(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, slice, slice, np.ndarray]:
    full_img = np.full_like(image, 255, shape=(991, 400, 3))
    full_seg = np.full_like(label, 0, shape=(991, 400), dtype=np.uint8)

    startx = (400 - image.shape[1]) // 2
    starty = (991 - image.shape[0]) // 2

    slicex = slice(startx, startx + image.shape[1])
    slicey = slice(starty, starty + image.shape[0])

    full_img[slicey, slicex] = image
    full_seg[slicey, slicex] = label + 1
    # full_seg[slicey.start] = 0
    # full_seg[slicey.stop-1] = 0

    return full_img, full_seg, slicex, slicey

class TumorBoi(nn.Module):

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        self.processor = AutoImageProcessor.from_pretrained(
            self.config.pretrain_path,
            ignore_index=255,
            reduce_labels=True,
            image_mean=[0.8630414518385323, 0.8630414518385329, 0.8630414518385320],
            image_std=[0.2181276311793262, 0.2181276311793267, 0.2181276311793268],
        )
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.config.pretrain_path,
            config = self.config.config,
            ignore_mismatched_sizes = True,
        )

    def forward(self, images: list[np.ndarray], labels: list[np.ndarray]):
        slices = list()
        images = images.copy()
        labels = labels.copy()
        for i in range(len(images)):
            images[i], labels[i], slicex, slicey = pad(images[i], labels[i])
            slices.append((slicex, slicey))
        inputs = self.processor.preprocess(images, labels, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        pixel_mask = inputs['pixel_mask'].to(device)
        mask_labels = [x.to(device) for x in inputs['mask_labels']]
        class_labels = [x.to(device) for x in inputs['class_labels']]

        for i in range(len(images)):
            pixel_mask[i].zero_()
            pixel_mask[i, slices[i][1], slices[i][0]] = 1

        out = self.mask2former(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            pixel_mask=pixel_mask,
        )
        setattr(out, "slices", slices)
        return out

    def out_to_seg(self, out) -> np.ndarray:
        slicex, slicey = out.slices[0]
        seg = self.processor.post_process_semantic_segmentation(out)[0].cpu().numpy().astype(np.uint8) - 1
        seg = cv2.resize(seg, (slicex.stop-slicex.start, slicey.stop-slicey.start)).astype(bool)[slicey, slicex]
        return seg

    def out_to_segs(self, out) -> list[np.ndarray]:
        slices = out.slices
        segs = self.processor.post_process_semantic_segmentation(out)
        for i, seg in enumerate(segs):
            slicex, slicey = slices[i]
            seg = seg.cpu().numpy().astype(np.uint8)
            seg = cv2.resize(seg, (400, 991)).astype(bool)[slicey, slicex]
            segs[i] = seg

        return segs
