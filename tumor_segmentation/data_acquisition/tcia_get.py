from os import PathLike
from pathlib import Path
from tcia_utils import nbia
import requests
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_scan(path: Path):
    slices = [pydicom.read_file(file) for file in path.iterdir() if str(file).endswith(".dcm")]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(scans, label=False):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    image[image == -2000] = 0

    for slice_number in range(len(scans)):
        intercept = scans[slice_number].RescaleIntercept
        slope = scans[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    norm_img = (np.array(image) / (2**16 - 1)).max(0)
    if label:
        # TODO: Match dataset
        norm_img[norm_img > 0] = 1.0
        breakpoint()
        return norm_img
    else:
        return 1 - norm_img


def download(collection: str, path: Path, label=False, number=0):
    data_series = nbia.getSeries(collection=collection, modality="RTSTRUCT" if label else "PT")
    nbia.downloadSeries(data_series, path=str(path), number=number)

def convert_to_imgs(path: Path, out_path: Path, label=False):
    out_path.mkdir(parents=True, exist_ok=True)
    for i, img_dir in enumerate(path.iterdir()):
        scans = load_scan(img_dir)
        img = get_pixels_hu(scans)
        plt.imsave(out_path / f"{'segmentation' if label else 'patient'}_{i}.png", img, cmap='gray')


if __name__ == "__main__":
    collection = "CPTAC-SAR"
    number = 10

    path = Path("local-data") / collection.lower()
    download(collection, path / "download", number=number)
    convert_to_imgs(path / "download", path / "imgs")

    # download(path / "download-label", label=True)
    # convert_to_imgs(path / "download-label", path / "labels", label=True)
