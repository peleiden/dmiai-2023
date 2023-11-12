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

def load_data(control_files: list[str], patient_files: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
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

    for i, (im, seg) in enumerate(zip(images, segmentations, strict=True)):
        # Jeg stoler ikke på noget
        assert im.shape == seg.shape, f"{im.shape = } {seg.shape}"
        assert (im[..., 0] == im[..., 1]).all()
        assert (im[..., 0] == im[..., 2]).all()
        assert (seg[..., 0] == seg[..., 1]).all()
        assert (seg[..., 0] == seg[..., 2]).all()
        segmentations[i] = seg[..., 0]

    return images, segmentations

def _sel(arrays: list[np.ndarray], idx: np.ndarray) -> list[np.ndarray]:
    return [arrays[i] for i in idx]

def split_train_test(images: list[np.ndarray], segmentations: list[np.ndarray], train_cfg: TrainConfig, n_control: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    control_images, images = images[:n_control], images[n_control:]
    control_segmentations, segmentations = segmentations[:n_control], segmentations[n_control:]
    n = len(images)
    index = np.arange(n)
    np.random.shuffle(index)  # inplace >:(
    n_train = int(train_cfg.train_test_split * n)

    train_images = control_images + _sel(images, index[:n_train])
    train_segmentations = control_segmentations + _sel(segmentations, index[:n_train])

    test_images = _sel(images, index[n_train:])
    test_segmentations = _sel(segmentations, index[n_train:])
    return train_images, train_segmentations, test_images, test_segmentations

def dataloader(train_cfg: TrainConfig, images: list[np.ndarray], segmentations: list[np.ndarray], augmentations=None):
    assert len(images) == len(segmentations)
    n = len(images)
    while True:
        index = np.random.randint(0, n, train_cfg.batch_size)
        batch_images = _sel(images, index)
        batch_segmentations = _sel(segmentations, index)
        if augmentations is not None:
            for i in range(train_cfg.batch_size):
                # Albuementations wants ints on boffa them
                augmented = augmentations(image=batch_images[i], mask=(batch_segmentations[i] * 255))
                batch_images[i] = augmented['image']
                batch_segmentations[i] = augmented['mask'].astype(bool)
        yield batch_images, batch_segmentations

def dice(targets: list[np.ndarray], preds: list[np.ndarray]) -> float:
    dice_bois = []
    for target, pred in zip(targets, preds, strict=True):
        if target.sum() == 0 and pred.sum() == 0:
            return 1
        tp = (target & pred).sum()
        fp = (~target & pred).sum()
        fn = (target & ~pred).sum()
        d = 2 * tp / (2 * tp + fp + fn)
        assert not np.isnan(d), breakpoint()
        dice_bois.append(d)
    return float(np.mean(dice_bois))

def vote(pred_segs: list[list[np.ndarray]]) -> list[np.ndarray]:
    voted_segs = []
    for pred_seg in pred_segs:
        voted_segs.append(np.round(np.array(pred_seg).mean(axis=0)).astype(bool))
    return voted_segs

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
