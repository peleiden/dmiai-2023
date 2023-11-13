from dataclasses import dataclass

import numpy as np
import torch
from pelutils import DataStorage


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainConfig(DataStorage):

    lr: float = 1e-5
    batch_size: int = 20
    batches: int = 500
    num_models: int = 3
    pretrain_path: str = "facebook/mask2former-swin-large-ade-semantic"
    splits: int = 5
    train_control_prob: float = 0.5
    dropout: float = 0.0

    def __post_init__(self):
        assert self.num_models % 2 == 1

    @property
    def config(self) -> str:
        return self.pretrain_path.replace("facebook/", "tumor_segmentation/configs/") + ".json"

@dataclass
class TrainResults(DataStorage):

    test_batches:  list[int]
    train_loss:    list[list[float]]  # Batch, model
    test_loss:     list[list[float]]
    train_dice:    list[list[float]]
    test_dice:     list[list[float]]
    ensemble_dice: list[float]

    @classmethod
    def empty(cls, train_cfg: TrainConfig):
        return TrainResults(
            test_batches  = list(),
            train_loss    = [list() for _ in range(train_cfg.num_models)],
            test_loss     = [list() for _ in range(train_cfg.num_models)],
            train_dice    = [list() for _ in range(train_cfg.num_models)],
            test_dice     = [list() for _ in range(train_cfg.num_models)],
            ensemble_dice = list(),
        )
