import base64
import json
import random
from functools import wraps
from typing import Callable

import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log

from lunar_lander.run_agent import load_agent


app = Flask(__name__)
Api(app)
CORS(app)

def _get_data() -> tuple[list[float], float, bool, float, int]:
    """Returns data from a post request"""
    data = json.loads(request.data.decode("utf-8"))
    return data["observation"], data["reward"], data["is_terminal"], data["total_reward"], data["game_ticks"]

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
    obs, reward, is_terminal, total_reward, game_ticks = _get_data()
    action = agent.act(obs)
    return { "action": action }

if __name__ == "__main__":
    log.configure(
        "lunar.log",
        append=True,
    )
    agent = load_agent("lunar_lander/agent_dqn_7.tar", True)
    app.run(host="0.0.0.0", port=6971, debug=False, processes=1, threaded=False)
