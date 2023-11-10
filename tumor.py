import base64
import json
from functools import wraps
from typing import Callable

import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log


app = Flask(__name__)
Api(app)
CORS(app)

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

@app.route("/predict", methods=["POST"])
@api_fun
def predict():
    data = _get_data()
    print(data)
    return { "img": [0] * len(data) }

if __name__ == "__main__":
    log.configure(
        "tumor.log",
        append=True,
    )
    app.run(host="0.0.0.0", port=6970, debug=False, processes=1, threaded=False)
