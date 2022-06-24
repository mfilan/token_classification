import unittest
from pathlib import Path
from typing import List, Dict

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from parameterized import parameterized  # type: ignore

from src.data import transform


def prepare_config():
    GlobalHydra.instance().clear()
    initialize(version_base="1.2", config_path="conf")
    cfg = compose(config_name="config", return_hydra_config=True)
    HydraConfig().cfg = cfg
    OmegaConf.resolve(cfg)
    return cfg


class TestTextObject(unittest.TestCase):
    cfg: DictConfig
    text_dict: Dict
    image_height: int
    image_width: int
    text_object: transform.TextObject

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = prepare_config()
        cls.text_dict = {
            "box": [
                1,
                999,
                799,
                1000
            ],
            "text": "COMPOUND",
            "label": "question",
            "words": [
                {
                    "box": [
                        1,
                        999,
                        799,
                        1000
                    ],
                    "text": "COMPOUND"
                }
            ],
            "linking": [
                [
                    0,
                    37
                ]
            ],
            "id": 0
        }
        cls.image_height = 1000
        cls.image_width = 800
        cls.text_object = transform.TextObject(cls.text_dict['box'],
                                               cls.text_dict['text'],
                                               cls.text_dict['label'],
                                               cls.text_dict['words'],
                                               cls.text_dict['id'],
                                               cls.image_height,
                                               cls.image_width)

    @parameterized.expand(
        [
            ([1, 999, 799, 1000], 800, 1000, [1, 999, 998, 1000])
        ]
    )
    def test_normalize_box(self, box: List[int], image_width: int, image_height: int, expected: List[int]) -> None:
        assert self.text_object.normalize_box(box, image_width, image_height) == expected

    @parameterized.expand(
        [
            (0, 1, 'S-'),
            (0, 0, 'B-'),
            (1, 2, 'E-'),
            (2, 4, 'I-'),
            (3, 3, 'O-'),
        ]
    )
    def test_get_label_prefix(self, word_index: int, num_words: int, expected: str) -> None:
        assert self.text_object.get_label_prefix(word_index, num_words) == expected


class TestTrainingDocumentWarehouse(unittest.TestCase):
    cfg: DictConfig
    training_document_warehouse: transform.TrainingDocumentWarehouse

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = prepare_config()
        cls.training_document_warehouse = transform.TrainingDocumentWarehouse(cls.cfg.dataset.images_dir,
                                                                              cls.cfg.dataset.annotations_dir)

    @parameterized.expand(
        [
            (Path('../datasets/FUNSD/images/82562350.png'), '82562350'),
            (Path('82562350.png'), '82562350'),
            (Path('images/lorem ipsum.png'), 'lorem ipsum'),
            (Path('files/82562350.tar.gz'), '82562350'),
            (Path('lorem ipsum/82562350.tar.gz'), '82562350')
        ]
    )
    def test_get_file_name(self, value: Path, expected: str) -> None:
        assert self.training_document_warehouse.get_file_name(value) == expected

    @parameterized.expand(
        [
            (
                    [Path('../datasets/FUNSD/annotations/82562350.json')],
                    [Path('../datasets/FUNSD/images/82562350.png')],
                    ['../datasets/FUNSD/annotations/82562350.json'],
                    ['../datasets/FUNSD/images/82562350.png']
            ),
            (
                    [Path('annotations/a.json')],
                    [Path('../datasets/FUNSD/images/a.png')],
                    ['annotations/a.json'],
                    ['../datasets/FUNSD/images/a.png']
            ),
            (
                    [Path('a.json'), Path('b.json'), Path('c.json')],
                    [Path('c.png'), Path('a.png'), Path('b.png')],
                    ['a.json', 'b.json', 'c.json'],
                    ['a.png', 'b.png', 'c.png']
            ),
            (
                    [Path('a.json'), Path('b.json'), Path('c.json'), Path('d.json')],
                    [Path('c.png'), Path('a.png'), Path('b.png'), Path('e.png')],
                    ['a.json', 'b.json', 'c.json'],
                    ['a.png', 'b.png', 'c.png']
            ),
            (
                    [], [], [], []
            ),
            (
                    ['a.json'], [], [], []
            ),
            (
                    [], ['a.png'], [], []
            ),
        ]
    )
    def test_pair_up_files(
            self,
            annotation_file_paths: List[Path],
            image_file_paths: List[Path],
            expected_annotations: List[str],
            expected_images: List[str]
    ) -> None:
        assert self.training_document_warehouse.pair_up_files(
            annotation_file_paths, image_file_paths) == (expected_annotations, expected_images)
