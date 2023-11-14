from collections import deque

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from matplotlib.backend_bases import MouseButton

from tumor_segmentation.data import load_data, get_data_files
from tumor_segmentation.mask_to_border import mask_to_border


control_files, patient_files, extra_patient_files = get_data_files()
print(len(control_files),len(patient_files), len(extra_patient_files))
control_imgs, control_segs = load_data(control_files[:10], list(), list())
patient_imgs, patient_segs = load_data(list(), patient_files[:10], list())
print(len(control_imgs))
print(len(patient_imgs))

def get_augmentation_pipeline(p: float):
    return A.Compose([
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.Rotate(limit=10, p=p),
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
        # print(len(queue), len(visited))
        # if len(queue) > 10000:
        #     breakpoint()
        for neighbour in neighbours(point):
            if neighbour in visited:
                continue
            try:
                neighbour_pix = seg[neighbour]
            except IndexError:
                continue

            if neighbour_pix:
                # print(neighbour, neighbour_pix)
                visited.add(neighbour)
                queue.append(neighbour)

    return tuple(np.array(list(visited)).T)

binding_id = plt.connect('button_press_event', lambda event: None)
augmentations = get_augmentation_pipeline(p=0.2)
for i, (control_img, control_seg) in enumerate(zip(control_imgs, control_segs, strict=True)):
    plt.disconnect(binding_id)
    control_img_orig = control_img.copy()
    patient_idx = np.random.randint(len(patient_imgs))
    patient_img = patient_imgs[patient_idx]
    patient_seg = patient_segs[patient_idx]
    patient_border = mask_to_border(patient_seg)
    patient_img_show = patient_img.copy()
    patient_img_show[np.where(patient_border)] = (0, 153, 255)

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
                x_min = min(tumor_where[1])
                x_max = max(tumor_where[1])
                y_min = min(tumor_where[0])
                y_max = max(tumor_where[0])
                dx = x_max - x_min
                dy = y_max - y_min
                tumor_img = patient_img[y_min:y_max, x_min:x_max]
                tumor_seg = patient_seg[y_min:y_max, x_min:x_max]
                rots = np.random.randint(4)
                tumor_img = np.rot90(tumor_img, rots, (0, 1))
                tumor_seg = np.rot90(tumor_seg, rots)
                assert tumor_img.shape[:2] == tumor_seg.shape
                dy, dx = tumor_seg.shape
                augmented = augmentations(image=tumor_img, mask=tumor_seg * 255)
                tumor_img = augmented['image']
                tumor_seg = augmented['mask'].astype(bool)
                rots = np.random.randint(4)
                alpha = np.random.uniform(0.7, 1)
                control_img[y:y+dy, x:x+dx][np.where(tumor_seg)] = alpha * tumor_img[np.where(tumor_seg)] + (1 - alpha) * control_img[y:y+dy, x:x+dx][np.where(tumor_seg)]
                control_seg[y:y+dy, x:x+dx][np.where(tumor_seg)] = tumor_seg[np.where(tumor_seg)]

                control_img_blur = ndimage.gaussian_filter(control_img, sigma=2, truncate=4)
                control_img[y:y+dy, x:x+dx][np.where(tumor_seg)] = control_img_blur[y:y+dy, x:x+dx][np.where(tumor_seg)]

        plt.subplot(122)
        control_img_show = control_img.copy()
        control_img_show[np.where(mask_to_border(control_seg))] = (255, 153, 0)
        plt.imshow(control_img_show)
        plt.show()

    plt.subplot(121)
    plt.imshow(patient_img_show)

    plt.subplot(122)
    plt.imshow(control_img)

    binding_id = plt.connect('button_press_event', on_click)

    plt.show()
    break



# binding_id = plt.connect('motion_notify_event', on_move)
# plt.connect('button_press_event', on_click)

# plt.show()