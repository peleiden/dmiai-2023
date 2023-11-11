import base64
import json
import random
from functools import wraps
from typing import List

import numpy as np
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
from pelutils import log, TT
from pydantic import BaseModel

from lunar_lander.run_agent import load_agent


class LunarLanderPredictRequestDto(BaseModel):
    observation: List[float]
    reward: float
    is_terminal: bool
    total_reward: float
    game_ticks: int

class LunarLanderPredictResponseDto(BaseModel):
    action: int

# app = Flask(__name__)
# Api(app)
# CORS(app)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def _get_data() -> tuple[list[float], float, bool, float, int]:
#     """Returns data from a post request"""
#     data = json.loads(request.data.decode("utf-8"))
#     return data["observation"], data["reward"], data["is_terminal"], data["total_reward"], data["game_ticks"]

# def api_fun(func) -> Callable:
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         with log.log_errors:
#             log("Received call to %s" % func.__name__)
#             res = func(*args, **kwargs)
#             return jsonify(res)

#     return wrapper

router = APIRouter()


log.configure(
    "lunar.log",
    print_level=0,
)
agent = load_agent("lunar_lander/agent-dqn.pt", True)

@router.post('/predict', response_model=LunarLanderPredictResponseDto)
def predict(request: LunarLanderPredictRequestDto):
    global gameno
    obs = request.observation
    reward = request.reward
    is_terminal = request.is_terminal
    total_reward = request.total_reward
    game_ticks = request.game_ticks

    if obs[-2] and obs[-1]:
        action = 0
    else:
        action = agent.act(obs)

    log.debug(f"{gameno} {game_ticks:02} {action} {reward:2f} {total_reward:2f}")
    if is_terminal:
        gameno += 1
    return LunarLanderPredictResponseDto(action=action)
