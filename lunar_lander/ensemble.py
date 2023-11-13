from .agent_class import agent_base
from scipy.stats import mode


def act_ensemble(obs, agents: list[agent_base]) -> int:
    acts = [agent.act(obs) for agent in agents]
    return int(mode(acts).mode)
