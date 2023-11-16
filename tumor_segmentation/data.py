import os
from glob import glob as glob
from pathlib import Path  # glob

import cv2
import numpy as np
import albumentations as A
import h5py
from sklearn.model_selection import KFold

from tumor_segmentation import TrainConfig

def get_augmentation_pipeline(p: float):
    return A.Compose([
        A.HorizontalFlip(p=p),
        A.Rotate(limit=5, p=p),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=p),
        A.GaussNoise(var_limit=(10, 50), p=p),
        # TODO: Add cropping which removes part of inputs but requires reconsideration of padding
    ])

def get_data_files() -> tuple[list[str], list[str], list[str]]:
    control_files = glob("tumor_segmentation/data/controls-augment/imgs/**/*.png", recursive=True)
    patient_files = glob("tumor_segmentation/data/patients/imgs/**/*.png", recursive=True)
    extra_patient_files = glob("tumor_segmentation/data/kaggle-patients/imgs/**/*.png", recursive=True)
    return control_files, patient_files, extra_patient_files

def load_h5_archive(path: str):
    images, segmentations = [], []
    with h5py.File(path, "r") as file:
        for pet_data, label_data in  zip(file["pet_data"].values(), file["label_data"].values(), strict=True):
            images.append(1- np.sqrt(np.max(pet_data, 1).squeeze()[::-1,:]))
            segmentations.append(np.max(label_data, 1).squeeze()[::-1,:])
    return images, segmentations

def load_data(control_files: list[str], patient_files: list[str], extra_patient_files: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ Returns stacked images and stacked segmentations. """
    label_files_augment = [p.replace("/imgs/", "/labels/").replace("/patient_", "/segmentation_") for p in control_files]
    label_files = [p.replace("/imgs/", "/labels/").replace("/patient_", "/segmentation_") for p in patient_files + extra_patient_files]

    control_images = [cv2.imread(p)[..., ::-1] for p in control_files]
    patient_images = [cv2.imread(p)[..., ::-1] for p in patient_files + extra_patient_files]

    for p in label_files_augment:
        assert os.path.isfile(p), p
    control_segmentations = [cv2.imread(p)[..., ::-1] for p in label_files_augment]

    for p in label_files:
        assert os.path.isfile(p), p
    patient_segmentations = [cv2.imread(p)[..., ::-1] for p in label_files]

    images = control_images + patient_images
    segmentations = control_segmentations + patient_segmentations
    for seg in segmentations:
        seg[seg > 0] = 255

    for i, (im, seg) in enumerate(zip(images, segmentations, strict=True)):
        # Jeg stoler ikke på noget
        assert im.shape == seg.shape, f"{im.shape = } {seg.shape}"
        assert (im[..., 0] == im[..., 1]).all()
        assert (im[..., 0] == im[..., 2]).all()
        assert (seg[..., 0] == seg[..., 1]).all()
        assert (seg[..., 0] == seg[..., 2]).all()
        seg = seg[..., 0]
        assert ((seg == 0) | (seg == 255)).all()
        segmentations[i] = np.round(seg / 255).astype(bool)
    return images, segmentations

def _sel(arrays: list[np.ndarray], idx: np.ndarray) -> list[np.ndarray]:
    return [arrays[i] for i in idx]

def split_train_test(images: list[np.ndarray], segmentations: list[np.ndarray], train_cfg: TrainConfig, n_control: int, n_extra: int, split: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    control_images, images = images[:n_control], images[n_control:]
    images, extra_images = images[:-n_extra], images[-n_extra:]
    control_segmentations, segmentations = segmentations[:n_control], segmentations[n_control:]
    segmentations, extra_segmentations = segmentations[:-n_extra], segmentations[-n_extra:]
    n = len(images)
    index = np.arange(n)

    if train_cfg.splits > 1:
        np.random.seed(420)
        np.random.shuffle(index)  # inplace >:(
        kf = KFold(train_cfg.splits)

        train_index, test_index = list(kf.split(np.arange(n)))[split]

        train_images = extra_images + control_images + _sel(images, index[train_index])
        train_segmentations = extra_segmentations + control_segmentations + _sel(segmentations, index[train_index])

        test_images = _sel(images, index[test_index])
        test_segmentations = _sel(segmentations, index[test_index])
    else:
        train_images = extra_images + control_images + images
        train_segmentations = extra_segmentations + control_segmentations + segmentations
        test_images = list()
        test_segmentations = list()

    return train_images, train_segmentations, test_images, test_segmentations

def dataloader(train_cfg: TrainConfig, images: list[np.ndarray], segmentations: list[np.ndarray], augmentations=None, n_control=None,  is_test=False):
    assert len(images) == len(segmentations)
    n = len(images)
    while True:
        if is_test:
            index = np.random.choice(len(images), len(images), replace=False)
            yield _sel(images, index), _sel(segmentations, index)
            continue
        if n_control is None:
            index = np.random.randint(0, n, train_cfg.batch_size)
        else:
            control_batch = int(train_cfg.batch_size * train_cfg.train_control_prob)
            index = np.concatenate((np.random.randint(0, n_control, control_batch), np.random.randint(n_control, n, train_cfg.batch_size - control_batch)))
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
            dice_bois.append(1.0)
            continue
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
    # images, segmentations = load_h5_archive("local-data/lab_petct_vox_5.00mm.h5")

    # import matplotlib.pyplot as plt
    # import pelutils.ds.plots as plots

    # out_dir = Path("tumor_segmentation/data/kaggle-patients")
    # img_dir = out_dir / "imgs"
    # label_dir = out_dir / "labels"
    # for d in img_dir, label_dir:
    #     d.mkdir(parents=True, exist_ok=True)

    # for i, (image, segmentation) in enumerate(zip(images, segmentations, strict=True)):
    #     print(f"{image.shape = } {segmentation.shape = }")
    #     plt.imsave(img_dir / f"patient_{i}.png", image, cmap='gray')
    #     plt.imsave(label_dir / f"segmentation_{i}.png", segmentation, cmap='gray')

    #     with plots.Figure("tumor_segmentation/h5-samples/sample_%i.png" % i, figsize=(20, 8), tight_layout=False):
    #         plt.subplot(121)
    #         plt.imshow(image, cmap="gray")
    #         plt.title("H5 image")
    #         plt.subplot(122)
    #         plt.imshow(segmentation, cmap="gray")
    #         plt.title("H5 segmentation")
    control_files, patient_files, _ = get_data_files()
    images, segmentations = load_data(control_files, list(), list())
    print(f"{len(images) = } {len(segmentations) = }")
    # train_images, train_segmentations, test_images, test_segmentations = split_train_test(images, segmentations)
    # print(f"{len(train_images) = } {len(train_segmentations) = }")
    # print(f"{len(test_images) = } {len(test_segmentations) = }")

    import matplotlib.pyplot as plt
    import pelutils.ds.plots as plots
    from tumor_segmentation.mask_to_border import mask_to_border
    from tumor_segmentation.model import channel_fuckwy

    for i in range(5):
        with plots.Figure("tumor_segmentation/samples/sample_%i.png" % i, figsize=(15, 15), tight_layout=False):
            img = channel_fuckwy(images[i])
            plt.subplot(141)
            plt.imshow(img[..., 0])
            plt.title("Red")
            plt.subplot(142)
            plt.imshow(img[..., 1])
            plt.title("Green")
            plt.subplot(143)
            plt.imshow(img[..., 2])
            plt.title("Blue")
            plt.subplot(144)
            images[i][np.where(mask_to_border(segmentations[i]))] = (0, 200, 0)
            plt.imshow(images[i])
            plt.title("Train image %i" % i)
