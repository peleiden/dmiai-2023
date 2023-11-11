from dataclasses import dataclass

from pelutils import DataStorage
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainConfig(DataStorage):

    lr: float = 1e-5
    batch_size: int = 4
    batches: int = 1000
    pretrain_path: str = "facebook/mask2former-swin-small-ade-semantic"
    config: str = "tumor_segmentation/mask2former-swin-small-ade-semantic.json"
