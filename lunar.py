""" Kør med python -m uvicorn lunar:router --host 0.0.0.0 --port 6971 """
from typing import List

from fastapi import APIRouter
from pelutils import log, TT
from pydantic import BaseModel

from lunar_lander.run_agent import load_agent
from lunar_lander.ensemble import act_ensemble


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
agents = [
    load_agent("lunar_lander/agent-593-dqn.best.pt"),
    load_agent("lunar_lander/agent-380-dqn.best.pt"),
    load_agent("lunar_lander/agent-12-ddqn.best.pt"),
    # load_agent("lunar_lander/agent-997-dqn.best.pt"),
    # load_agent("lunar_lander/agent-371-ac.best.pt"),
    # load_agent("lunar_lander/agent-477-ac.best.pt"),
    # load_agent("lunar_lander/agent-227-dqn.best.pt"),
    # load_agent("lunar_lander/agent-350-ac.best.pt"),
    # load_agent("lunar_lander/agent-47-ac.best.pt"),
    # load_agent("lunar_lander/agent-568-ac.best.pt"),
    # load_agent("lunar_lander/agent-290-ac.best.pt"),
    # load_agent("lunar_lander/agent-170-ac.best.pt"),
    # load_agent("lunar_lander/agent-736-ac.best.pt"),
    # load_agent("lunar_lander/agent-25-dqn.best.pt"),
    # load_agent("lunar_lander/agent-918-dqn.best.pt"),
    # load_agent("lunar_lander/agent-289-ac.best.pt"),
    # load_agent("lunar_lander/agent-201-ac.best.pt"),
    # load_agent("lunar_lander/agent-869-dqn.best.pt"),
]

# 273.4932801063924 264.13 /work3/s183912/trained-agents/agent-869-dqn.best.pt
# 273.5941244145446 240.21 /work3/s183912/trained-agents/agent-201-ac.best.pt
# 273.7096354038865 222.54 /work3/s183912/trained-agents/agent-289-ac.best.pt
# 274.2654732961101 217.89 /work3/s183912/trained-agents/agent-918-dqn.best.pt
# 274.7081851457472 208.53 /work3/s183912/trained-agents/agent-25-dqn.best.pt
# 274.82997907659376 230.64 /work3/s183912/trained-agents/agent-736-ac.best.pt
# 275.0994971407531 226.46 /work3/s183912/trained-agents/agent-170-ac.best.pt
# 275.28195735582835 215.7 /work3/s183912/trained-agents/agent-290-ac.best.pt
# 275.41494137954265 228.98 /work3/s183912/trained-agents/agent-568-ac.best.pt
# 275.60280826156526 231.26 /work3/s183912/trained-agents/agent-47-ac.best.pt
# 276.4515062503196 257.11 /work3/s183912/trained-agents/agent-350-ac.best.pt
# 277.9209871650508 229.85 /work3/s183912/trained-agents/agent-227-dqn.best.pt
# 279.8083759246668 222.77 /work3/s183912/trained-agents/agent-477-ac.best.pt
# 279.8357025271986 220.92 /work3/s183912/trained-agents/agent-371-ac.best.pt
# 284.01782526577574 215.44 /work3/s183912/trained-agents/agent-997-dqn.best.pt

gameno = 0
@router.post('/predict', response_model=LunarLanderPredictResponseDto)
def predict(request: LunarLanderPredictRequestDto):
    global gameno
    obs = request.observation
    reward = request.reward
    is_terminal = request.is_terminal
    total_reward = request.total_reward
    game_ticks = request.game_ticks

    action = act_ensemble(obs, agents)

    if is_terminal:
        log.debug(f"{gameno} {game_ticks:02} {action} {reward:2f} {total_reward:2f}")
        gameno += 1
    return LunarLanderPredictResponseDto(action=action)
