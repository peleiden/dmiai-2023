import os
from glob import glob as glob  # glob

import cv2
import numpy as np
import albumentations as A

from tumor_segmentation import TrainConfig

def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.2),
        A.Rotate(limit=5, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        # TODO: Add cropping which removes part of inputs but requires reconsideration of padding
    ])

def get_data_files() -> tuple[list[str], list[str]]:
    control_files = glob("tumor_segmentation/data/controls/**/*.png", recursive=True)
    patient_files = glob("tumor_segmentation/data/patients/imgs/**/*.png", recursive=True)
    return control_files, patient_files

def load_data(control_files: list[str], patient_files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """ Returns stacked images and stacked segmentations. """
    label_files = [p.replace("/imgs/", "/labels/").replace("/patient_", "/segmentation_") for p in patient_files]

    control_images = [cv2.imread(p)[..., ::-1] for p in control_files]
    patient_images = [cv2.imread(p)[..., ::-1] for p in patient_files]

    control_segmentations = [np.zeros_like(ci) for ci in control_images]
    for p in label_files:
        assert os.path.isfile(p), p
    patient_segmentations = [cv2.imread(p)[..., ::-1] for p in label_files]

    images = control_images + patient_images
    segmentations = control_segmentations + patient_segmentations

    imarr = np.empty((len(images), 991, 400, 3), dtype=np.uint8)
    segarr = np.empty((len(images), 991, 400), dtype=np.uint8)

    for i, (im, seg) in enumerate(zip(images, segmentations, strict=True)):
        # Jeg stoler ikke på noget
        assert im.shape == seg.shape, f"{im.shape = } {seg.shape}"
        assert (im[..., 0] == im[..., 1]).all()
        assert (im[..., 0] == im[..., 2]).all()
        assert (seg[..., 0] == seg[..., 1]).all()
        assert (seg[..., 0] == seg[..., 2]).all()

        margin_x = (400 - im.shape[1]) // 2
        margin_y = (991 - im.shape[0]) // 2
        full_im = np.full((991, 400, 3), 255, dtype=np.uint8)
        full_seg = np.full((991, 400), 0, dtype=np.uint8)
        full_im[margin_y:margin_y+im.shape[0], margin_x:margin_x+im.shape[1]] = im
        full_seg[margin_y:margin_y+im.shape[0], margin_x:margin_x+im.shape[1]] = seg[..., 0]

        imarr[i] = full_im
        segarr[i] = full_seg

    assert ((segarr == 0) | (segarr == 255)).all()
    segarr = np.round(segarr / 255).astype(bool)

    return imarr, segarr

def split_train_test(images: np.ndarray, segmentations: np.ndarray, train_cfg: TrainConfig, n_control: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    control_images, images = images[:n_control], images[n_control:]
    control_segmentations, segmentations = segmentations[:n_control], segmentations[n_control:]
    n = len(images)
    index = np.arange(n)
    np.random.shuffle(index)  # inplace >:(
    n_train = int(train_cfg.train_test_split * n)

    train_images = np.concatenate([control_images, images[index[:n_train]]])
    train_segmentations = np.concatenate([control_segmentations, segmentations[index[:n_train]]])

    test_images = images[index[n_train:]]
    test_segmentations = segmentations[index[n_train:]]

    return train_images, train_segmentations, test_images, test_segmentations

def dataloader(train_cfg: TrainConfig, images: np.ndarray, segmentations: np.ndarray, augmentations=None):
    assert len(images) == len(segmentations)
    n = len(images)
    while True:
        index = np.random.randint(0, n, train_cfg.batch_size)
        index = np.random.randint(0, n, train_cfg.batch_size)
        batch_images = images[index]
        batch_segmentations = segmentations[index]
        if augmentations is not None:
            for i in range(train_cfg.batch_size):
                # Albuementations wants ints on boffa them
                augmented = augmentations(image=batch_images[i], mask=(batch_segmentations[i] * 255))
                batch_images[i] = augmented['image']
                batch_segmentations[i] = augmented['mask'].astype(bool)
        yield batch_images, batch_segmentations

def dice(target: np.ndarray, pred: np.ndarray) -> float:
    if target.sum() == 0 and pred.sum() == 0:
        return 1
    tp = (target & pred).sum()
    fp = (~target & pred).sum()
    fn = (target & ~pred).sum()
    d = 2 * tp / (2 * tp + fp + fn)
    assert not np.isnan(d), breakpoint()
    return d

def vote(pred_segs: np.ndarray) -> np.ndarray:
    return np.round(pred_segs.mean(axis=0)).astype(bool)

if __name__ == "__main__":
    control_files, patient_files = get_data_files()
    images, segmentations = load_data(control_files, patient_files)
    print(f"{images.shape = } {segmentations.shape = }")
    train_images, train_segmentations, test_images, test_segmentations = split_train_test(images, segmentations)
    print(f"{train_images.shape = } {train_segmentations.shape = }")
    print(f"{test_images.shape = } {test_segmentations.shape = }")

    import matplotlib.pyplot as plt
    import pelutils.ds.plots as plots

    for i in range(5):
        with plots.Figure("tumor_segmentation/samples/sample_%i.png" % i, figsize=(20, 8), tight_layout=False):
            plt.subplot(141)
            plt.imshow(train_images[i])
            plt.title("Train image %i" % i)
            # plt.gca().set_aspect("equal")
            plt.subplot(142)
            plt.imshow(train_segmentations[i], cmap="gray")
            plt.title("Train segmentation %i" % i)
            # plt.gca().set_aspect("equal")
            plt.subplot(143)
            plt.imshow(test_images[i])
            plt.title("Test image %i" % i)
            # plt.gca().set_aspect("equal")
            plt.subplot(144)
            plt.imshow(test_segmentations[i], cmap="gray")
            plt.plot([1,2,3])
            plt.title("Test segmentation %i" % i)
            # plt.gca().set_aspect("equal")
