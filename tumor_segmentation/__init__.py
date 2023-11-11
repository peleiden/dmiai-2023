from dataclasses import dataclass

import numpy as np
import torch
from pelutils import DataStorage


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainConfig(DataStorage):

    lr: float = 1e-4
    batch_size: int = 16
    batches: int = 300
    num_models: int = 3
    pretrain_path: str = "facebook/mask2former-swin-small-ade-semantic"
    config: str = "tumor_segmentation/mask2former-swin-small-ade-semantic.json"
    train_test_split: float = 0.75

    def __post_init__(self):
        assert self.num_models % 2 == 1

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
