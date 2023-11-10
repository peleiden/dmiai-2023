import base64
import json
from functools import wraps
from typing import Callable

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log


app = Flask(__name__)
Api(app)
CORS(app)

def validate_segmentation(pet_mip, seg_pred):
    assert isinstance(seg_pred, np.ndarray), "Segmentation was not succesfully decoded as a numpy array"
    assert pet_mip.shape == seg_pred.shape, f"Segmentation of shape {seg_pred.shape} is not identical to image shape {pet_mip.shape}"

    unique_vals = list(np.unique(seg_pred))
    allowed_vals = [0, 255]
    unique_vals_str = ", ".join([str(x) for x in (unique_vals)])
    all_values_are_allowed = all(
        x in allowed_vals for x in unique_vals)
    assert all_values_are_allowed,  f"The segmentation contains values {{{unique_vals_str}}} but only values {{0,255}} are allowed"

    assert np.all(seg_pred[:, :, 0] == seg_pred[:, :, 1]) & np.all(
        seg_pred[:, :, 1] == seg_pred[:, :, 2]), "The segmentation values should be identical along the 3 color channels."

def encode_request(np_array: np.ndarray) -> str:
    # Encode the NumPy array as a png image
    success, encoded_img = cv2.imencode('.png', np_array)

    if not success:
        raise ValueError("Failed to encode the image")

    # Convert the encoded image to a base64 string
    base64_encoded_img = base64.b64encode(encoded_img.tobytes()).decode()

    return base64_encoded_img

def decode_request(b64im) -> np.ndarray:
    np_img = np.fromstring(base64.b64decode(b64im), np.uint8)
    a = cv2.imdecode(np_img, cv2.IMREAD_ANYCOLOR)
    return a

def _get_data():
    """Returns data from a post request"""
    data = json.loads(request.data.decode("utf-8"))
    return decode_request(data["img"])

def api_fun(func) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        with log.log_errors:
            log("Received call to %s" % func.__name__)
            res = func(*args, **kwargs)
            return jsonify(res)

    return wrapper

def threshold_classifier(img: np.ndarray) -> np.ndarray:
    return img < 50 / 255

def seg_to_shitty_rgb(seg: np.ndarray) -> np.ndarray:
    return np.stack((seg, seg, seg), axis=-1).astype(np.uint8) * 255

@app.route("/predict", methods=["POST"])
@api_fun
def predict():
    img_orig = _get_data()
    assert (img_orig[..., 0] == img_orig[..., 1]).all()
    assert (img_orig[..., 0] == img_orig[..., 2]).all()
    img = img_orig[..., 0] / 255
    # Replace this call
    seg = threshold_classifier(img)
    seg = seg_to_shitty_rgb(seg)
    validate_segmentation(img_orig, seg)
    return { "img": encode_request(seg) }

if __name__ == "__main__":
    log.configure(
        "tumor.log",
        append=True,
    )
    app.run(host="0.0.0.0", port=6970, debug=False, processes=1, threaded=False)
