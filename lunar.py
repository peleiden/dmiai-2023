""" Kør med python -m uvicorn lunar:router --host 0.0.0.0 --port 6971 """
from typing import List

from fastapi import APIRouter
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
