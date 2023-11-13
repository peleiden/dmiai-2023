from glob import glob as glob  # glob

import cv2
import numpy as np


def circular_kernel(diameter: int) -> np.ndarray:
    ''' Returns a circular kernel, that has ones on the circle and zeros outside. '''
    c = diameter / 2 - 0.5, diameter / 2 - 0.5

    Y, X = np.ogrid[:diameter, :diameter]

    dist_from_center = np.sqrt((X - c[0])**2 + (Y-c[1])**2)
    mask = dist_from_center <= diameter / 2
    return mask.astype(np.uint8)

def mask_to_border(mask: np.ndarray, thiccness=2, padding=5) -> np.ndarray:
    ''' Makes bounding borders around objects in the mask, which is a boolean matrix.
    Thiccness controls border with, and padding is the spacing between the objects and
    the corresponding borders. '''
    border_kernel = circular_kernel(2 * thiccness + 1)
    padding_kernel = circular_kernel(2 * padding + 1)
    thicc_mask = cv2.dilate(mask.astype(np.uint8), padding_kernel)
    thiccer_mask = cv2.dilate(thicc_mask, border_kernel)
    return np.logical_xor(thicc_mask.astype(bool), thiccer_mask.astype(bool))
