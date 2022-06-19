from dataclasses import dataclass


@dataclass
class Paths:
    data_dir: str
    test_data_annotations: str
    test_data_images: str
    train_data_annotations: str
    train_data_images: str


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
    paths: Paths
    params: Params
    model: Model
