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
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.sequence = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.sequence(x)
        
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.sequence(x)
        
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        #input_dim = 256
        self.encoder = nn.ModuleList([
            DownConv(in_channels, 64), #128
            DownConv(64, 128), #64
            DownConv(128, 256), #32
            DownConv(256, 512) #16
        ])
        
        self.bottleneck = ConvBlock(512, 1024)
        
        #extra channels allow for concatenation of skip connections in upsampling block
        self.decoder = nn.ModuleList([
            UpConv(512+1024,512), #32
            UpConv(256+512,256), #64
            UpConv(128+256,128), #128
            UpConv(64+128,64) #256
        ])
        
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        skips = []
        o = x
        for layer in self.encoder:
            o = layer(o)
            skips.append(o)
        
        o = self.bottleneck(o)
        
        for i, layer in enumerate(self.decoder):
            #print(o.size())
            o = torch.cat((skips[len(skips)-i-1],o), dim=1)
            #print(o.size())
            o = layer(o)
        
        return self.output_conv(o)
class UNETTTT(nn.Module):

    class Out:
        def __init__(self, loss, segs):
            self.loss = loss
            self.segs = segs

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config

        # self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #         in_channels=3, out_channels=1, init_features=32, pretrained=True)
        self.model = UNet()
        self.loss_fn = BinaryDiceLoss()
        self.bnorm = nn.BatchNorm2d(3)

    def forward(self, images: list[np.ndarray], labels: list[np.ndarray]):
        images = np.array([cv2.resize(im, (256, 256)) for im in images]) / 255
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device)
        print("Forwarding")
        preds = self.model(self.bnorm(images))[:, 0]
        print("Done forwarding", preds.mean())
        labels = np.array([cv2.resize(lab.astype(np.uint8), (256, 256)) for lab in labels])
        labels = torch.from_numpy(labels).float().to(device)

        print("lossy")
        out = self.Out(self.loss_fn(preds, labels), preds.detach().cpu().numpy())
        print("out")

        return out

    def out_to_seg(self, out, img: np.ndarray) -> np.ndarray:
        seg = cv2.resize(out.segs[0], (img.shape[1], img.shape[0]))
        seg = np.round(seg).astype(bool)
        return seg

    def out_to_segs(self, out, original_shapes) -> list[np.ndarray]:
        segs = list()
        for i, seg in enumerate(out.segs):
            seg = cv2.resize(seg, (original_shapes[i][::-1]))
            seg = np.round(seg).astype(bool)
            segs.append(seg)
        return segs
