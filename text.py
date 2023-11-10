import json
from functools import wraps
from typing import Callable

from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log

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
    print(len(data))
    return { "class_ids": [0] * len(data) }


if __name__ == "__main__":
    log.configure(
        "text.log",
        append=True,
    )
    app.run(host="0.0.0.0", port=6969, debug=False, processes=1, threaded=False)
