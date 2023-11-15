import os
from collections import deque

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from matplotlib.backend_bases import MouseButton

from tumor_segmentation.data import load_data, get_data_files
from tumor_segmentation.mask_to_border import mask_to_border, circular_kernel


control_files, patient_files, extra_patient_files = get_data_files()
print(len(control_files),len(patient_files), len(extra_patient_files))
control_imgs, control_segs = load_data(control_files, list(), list())
patient_imgs, patient_segs = load_data(list(), patient_files, list())

def seg_to_shitty_rgb(seg: np.ndarray) -> np.ndarray:
    return np.stack((seg, seg, seg), axis=-1).astype(np.uint8) * 255

def get_augmentation_pipeline(p: float):
    return A.Compose([
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.Rotate(limit=90, p=p),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p),
    ])

def tumor_clip(x: int, y: int, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not seg[y, x]:
        return (list(), list())

    start = (y, x)
    queue = deque([start])
    visited = { start }

    def neighbours(point: tuple[int, int]) -> list[tuple[int, int]]:
        return [
            (point[0] - 1, point[1]),
            (point[0] + 1, point[1]),
            (point[0], point[1] - 1),
            (point[0], point[1] + 1),
        ]

    while queue:
        point = queue.popleft()
        for neighbour in neighbours(point):
            if neighbour in visited:
                continue
            try:
                neighbour_pix = seg[neighbour]
            except IndexError:
                continue

            if neighbour_pix:
                visited.add(neighbour)
                queue.append(neighbour)

    return tuple(np.array(list(visited)).T)

binding_id = plt.connect('button_press_event', lambda event: None)
augmentations = get_augmentation_pipeline(p=0.2)
for i, (control_path, control_img, control_seg) in enumerate(zip(control_files, control_imgs, control_segs, strict=True)):
    outpath_img = control_path.replace("/controls/", "/controls-augment/").replace("control_", "patient_")
    outpath_seg = control_path.replace("/controls/", "/controls-augment/").replace("control_", "segmentation_").replace("/imgs/", "/labels/")
    if os.path.isfile(outpath_img) and os.path.isfile(outpath_seg):
        continue

    plt.disconnect(binding_id)
    control_img_orig = control_img.copy()
    patient_idx = np.random.randint(len(patient_imgs))
    patient_img = patient_imgs[patient_idx]
    patient_seg = patient_segs[patient_idx]
    patient_border = mask_to_border(patient_seg, padding=3)
    patient_img_show = patient_img.copy()
    patient_img_show[np.where(patient_border)] = (0, 153, 255)
    print(control_path)

    tumor_where = (list(), list())

    def on_click(event):
        global tumor_where, control_img
        if not event.inaxes:
            return
        if event.button is MouseButton.RIGHT:
            print("Reset")
            control_img = control_img_orig.copy()
            control_seg[...] = 0
        else:
            is_reference = event.inaxes._position._points[0][0] < 0.5
            x = int(event.xdata)
            y = int(event.ydata)

            if is_reference:
                tumor_where = tumor_clip(x, y, patient_seg)
            else:
                if len(tumor_where[0]) == 0:
                    print("No tumor selected")
                    return
                margin = 5
                x_min = min(tumor_where[1]) - margin
                x_max = max(tumor_where[1]) + margin
                y_min = min(tumor_where[0]) - margin
                y_max = max(tumor_where[0]) + margin
                dx = x_max - x_min
                dy = y_max - y_min

                # Grab tumor cutout
                tumor_img = patient_img[y_min:y_max, x_min:x_max].copy()
                tumor_seg = patient_seg[y_min:y_max, x_min:x_max].copy()
                tum = tumor_clip(dx // 2, dy // 2, tumor_seg)
                nottum = tumor_clip(dx // 2, dy // 2, ~tumor_seg)
                tumor_seg[tum] = 1
                tumor_seg[nottum] = 0

                # Do augmentations
                augmented = augmentations(image=tumor_img, mask=tumor_seg * 255)
                tumor_img = augmented['image']
                tumor_seg = augmented['mask'].astype(bool)

                xscale = np.random.uniform(0.8, 1.2)
                yscale = np.random.uniform(0.8, 1.2)
                tumor_img = cv2.resize(tumor_img, None, fx=xscale, fy=yscale)
                tumor_seg = cv2.resize(tumor_seg.astype(np.uint8), None, fx=xscale, fy=yscale).astype(bool)

                dy, dx = tumor_seg.shape

                # Paste stuff
                control_img[y:y+dy, x:x+dx][np.where(tumor_seg)] = tumor_img[np.where(tumor_seg)]
                control_seg[y:y+dy, x:x+dx][np.where(tumor_seg)] = np.minimum(tumor_seg[np.where(tumor_seg)], control_seg[y:y+dy, x:x+dx][np.where(tumor_seg)])

                control_img_blur = ndimage.gaussian_filter(control_img, sigma=np.random.uniform(0.5, 1), truncate=2)
                kernel = circular_kernel(5)
                where_blur = np.logical_xor(
                    cv2.dilate(tumor_seg.astype(np.uint8), kernel).astype(bool),
                    cv2.erode(tumor_seg.astype(np.uint8), kernel).astype(bool),
                )
                control_img[y:y+dy, x:x+dx][where_blur] = control_img_blur[y:y+dy, x:x+dx][where_blur]

        plt.subplot(122)
        control_img_show = control_img.copy()
        control_img_show[np.where(mask_to_border(control_seg, padding=3))] = (255, 153, 0)
        plt.imshow(control_img_show)
        plt.title(os.path.basename(control_path))
        plt.show()

    plt.subplot(121)
    plt.imshow(patient_img_show)

    plt.subplot(122)
    plt.imshow(control_img)
    plt.title(os.path.basename(control_path))

    binding_id = plt.connect('button_press_event', on_click)

    plt.show()

    cv2.imwrite(outpath_img, control_img)
    cv2.imwrite(outpath_seg, seg_to_shitty_rgb(control_seg))
