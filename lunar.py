import base64
import json
import random
from functools import wraps
from typing import Callable

import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log, TT

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

d = [list()]
gameno = 0
@app.route("/predict", methods=["POST"])
@api_fun
def predict():
    global gameno
    obs, reward, is_terminal, total_reward, game_ticks = _get_data()
    if obs[-2] and obs[-1]:
        action = 0
    else:
        action = agent.act(obs)
    d[-1].append({
        "obs": obs,
        "reward": reward,
        "total_reward": total_reward,
        "game_ticks": game_ticks,
        "action": action,
    })
    log.debug(f"{game_ticks:02} {action} {reward:2f} {total_reward:2f}")
    if is_terminal:
        with open("json_%i.json" % gameno, "w") as f:
            import json
            json.dump(d[-1], f, indent=4)
        gameno += 1
        d.append(list())
    return { "action": action }

if __name__ == "__main__":
    log.configure(
        "lunar.log",
        print_level=0,
    )
    agent = load_agent("lunar_lander/agent-dqn.pt", True)
    app.run(host="0.0.0.0", port=6971, debug=False, processes=1, threaded=False)
