import os
import json
from functools import wraps
from typing import Callable
from glob import glob as glob # glob

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import Parser, log
import spacy

from ai_text_detector.training.eval import Classifier

# python -m spacy download da_core_news_sm
nlp = spacy.load("da_core_news_sm")
app = Flask(__name__)
Api(app)
CORS(app)


def _get_data():
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

@app.route("/predict", methods=["POST"])
@api_fun
def predict():
    data = _get_data()
    preds = MODEL.predict(data, BATCH_SIZE)
    # FIXME FIXME FIXME REMOVE!
    preds = [1 - p for p in preds]
    # preds = [0] * len(_get_data())
    # with open("idx_to_test.int", "r", encoding="utf-8") as file:
    #     idx_to_test = int(file.read())
    # preds[idx_to_test] = 1
    #middle_boi = sorted(lens := [len(s.split()) for s in data])[len(data) // 2]
    #preds = [1 if abe > middle_boi else 0 for abe in lens]
    ##with open("cached_res.json", "r", encoding="utf-8") as file:
    ##    preds = json.load(file)
    #for i, ex in enumerate(data):
    #    if "vigtigt at huske" in ex:
    #        preds[i] = 1
    #    if "..." in ex:
    #        preds[i] = 0
    return {"class_ids": preds}


if __name__ == "__main__":
    location = Parser().parse_args().location
    MODEL = Classifier(glob(os.path.join(location, "dfm-encoder-large-v1-idx*-ai-detector")))
    BATCH_SIZE = 8
    log.configure(
        "text.log",
        append=True,
    )
    app.run(host="0.0.0.0", port=6969, debug=False, processes=1, threaded=False)
