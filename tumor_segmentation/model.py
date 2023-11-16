import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoConfig, AutoModelForSemanticSegmentation

from . import device, TrainConfig
from tumor_segmentation.dice import BinaryDiceLoss


def pad(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, slice, slice]:
    full_img = np.full_like(image, 255, shape=(991, 400, 3))
    full_seg = np.full_like(label, False, shape=(991, 400))

    startx = (400 - image.shape[1]) // 2
    starty = (991 - image.shape[0]) // 2

    slicex = slice(startx, startx + image.shape[1])
    slicey = slice(starty, starty + image.shape[0])

    full_img[slicey, slicex] = image
    full_seg[slicey, slicex] = label

    return full_img, full_seg, slicex, slicey

def channel_fuckwy(img: np.ndarray) -> np.ndarray:
    img = img.copy()
    img[..., 1] = 255 * (img[..., 0] < 80)
    img[..., 2] = cv2.Laplacian(img[..., 0], ddepth=-1, ksize=3, scale=2)
    return img

class TumorBoi(nn.Module):

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        self.processor = AutoImageProcessor.from_pretrained(self.config.pretrain_path)
        config = AutoConfig.from_pretrained(self.config.pretrain_path)
        config.id2label = {0: "healthy", 1: "tumor"}
        config.label2id = {"healthy": 0, "tumor": 1}
        self.mask2former = AutoModelForSemanticSegmentation.from_pretrained(
            self.config.pretrain_path,
            config = config,
            ignore_mismatched_sizes = True,
        )
        self._set_dropouts(self.config.dropout)

    def forward(self, images: list[np.ndarray], labels: list[np.ndarray]):
        slices = list()
        images = images.copy()
        labels = labels.copy()
        for i in range(len(images)):
            images[i], labels[i], slicex, slicey = pad(images[i], labels[i])
            slices.append((slicex, slicey))
        inputs = self.processor.preprocess(images, labels, return_tensors="pt").to(device)
        out = self.mask2former(**inputs)
        setattr(out, "slices", slices)
        return out

    def out_to_seg(self, out) -> np.ndarray:
        slicex, slicey = out.slices[0]
        seg = self.processor.post_process_semantic_segmentation(out)[0].cpu().numpy().astype(np.uint8)
        seg = cv2.resize(seg, (400, 991)).astype(bool)[slicey, slicex]
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

    def _set_dropouts(self, p: float):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = p
