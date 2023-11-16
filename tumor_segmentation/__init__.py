from __future__ import annotations

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
    pretrain: bool = True

    # For backwards compatability
    train_test_split: None = None

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

    @classmethod
    def mean(cls, *train_res: TrainResults) -> TrainResults:
        train_loss    = np.array([tr.train_loss for tr in train_res]).mean(axis=0).tolist()
        test_loss     = np.array([tr.test_loss for tr in train_res]).mean(axis=0).tolist()
        train_dice    = np.array([tr.train_dice for tr in train_res]).mean(axis=0).tolist()
        test_dice     = np.array([tr.test_dice for tr in train_res]).mean(axis=0).tolist()
        ensemble_dice = np.array([tr.ensemble_dice for tr in train_res]).mean(axis=0).tolist()
        return cls(
            train_res[0].test_batches,
            train_loss,
            test_loss,
            train_dice,
            test_dice,
            ensemble_dice,
        )
