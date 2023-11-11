from dataclasses import dataclass

from pelutils import DataStorage


@dataclass
class TrainConfig(DataStorage):

    lr: float = 1e-5
    batch_size: int = 8
    batches: int = 1000
    pretrain_path: str = "facebook/mask2former-swin-small-ade-semantic"
    config: str = "tumor_segmentation/mask2former-swin-small-ade-semantic.json"
