import json
import os
import time
from functools import wraps
from typing import Callable
from glob import glob as glob # glob

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import Parser, log

from ai_text_detector.training.eval import Classifier

app = Flask(__name__)
Api(app)
CORS(app)

USE_HPC = True
REVIEWS_TO_SEND = None
PREDICTIONS_RECEIVED = None
WAIT_TIME = 30

def _get_data() -> list[str]:
    """Returns data from a post request"""
    data = json.loads(request.data.decode("utf-8"))
    return data["answers"]

def api_fun(func) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        with log.log_errors:
            log("Received call to %s" % func.__name__)
            res = func(*args, **kwargs)
            return jsonify(res)

    return wrapper

@app.route("/to-hpc", methods=["GET"])
@api_fun
def to_hpc():
    global REVIEWS_TO_SEND
    if REVIEWS_TO_SEND is not None:
        texts = REVIEWS_TO_SEND
        REVIEWS_TO_SEND = None
        return texts
    else:
        return []

@app.route("/from-hpc", methods=["POST"])
@api_fun
def from_hpc():
    global PREDICTIONS_RECEIVED
    print("Receiving...")
    PREDICTIONS_RECEIVED = _get_data()

@app.route("/predict", methods=["POST"])
@api_fun
def predict():
    global REVIEWS_TO_SEND, PREDICTIONS_RECEIVED
    texts = _get_data()
    if USE_HPC:
        REVIEWS_TO_SEND = texts
        log.info("Sending reviews to HPC")
        waited_since = time.time()
        while PREDICTIONS_RECEIVED is None and (time.time() - waited_since < WAIT_TIME):
            pass
        if PREDICTIONS_RECEIVED is None:
            log.error("HPC did not respond.")
            preds = []
        else:
            log.info("HPC response!")
            preds = PREDICTIONS_RECEIVED
            PREDICTIONS_RECEIVED = None
    else:
        preds = MODEL.predict(texts, BATCH_SIZE)
    # FIXME FIXME FIXME REMOVE!
    preds = [1 - p for p in preds]
    return {"class_ids": preds}


if __name__ == "__main__":
    if not USE_HPC:
        location = Parser().parse_args().location
        MODEL = Classifier(glob(os.path.join(location, "dfm-encoder-large-v1*-ai-detector")))
        BATCH_SIZE = 64
    log.configure(
        "text.log",
        append=True,
    )
    app.run(host="0.0.0.0", port=6969, debug=False, processes=1, threaded=False)
