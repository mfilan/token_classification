from dataclasses import dataclass
from typing import Dict


@dataclass
class Dataset:
    data_dir: str
    annotations_dir: str
    images_dir: str
    label2id: Dict[str, int]
    id2label: Dict[int, str]


@dataclass
class Params:
    epoch_count: int
    lr: float
    batch_size: int


@dataclass
class Model:
    name: str
    num_classes: int


@dataclass
class NERConfig:
    dataset: Dataset
    params: Params
    model: Model
