import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Union, TypedDict

import numpy as np
import pandas as pd
from PIL import Image


class Datasets(TypedDict):
    train_dataset: Union[pd.DataFrame, None]
    test_dataset: Union[pd.DataFrame, None]
    validation_dataset: Union[pd.DataFrame, None]


@dataclass
class Word:
    box: List[int]
    normalized_box: List[int]
    text: str
    label: str


@dataclass
class TextObject:
    box: List[int]
    text: str
    label: str
    words_dict: List[Dict[str, Any]]
    id: int
    image_height: int
    image_width: int
    words: List[Word] = field(init=False, default_factory=list)
    normalized_box: List[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.normalized_box = self.normalize_box(self.box, self.image_width, self.image_height)
        for word_idx, word_dict in enumerate(self.words_dict):
            label_prefix = self.get_label_prefix(word_idx, len(self.words_dict))
            word_label = label_prefix + self.label if self.label != 'other' else self.label
            if word_dict['text'] != '':
                self.words.append(Word(word_dict['box'],
                                       self.normalize_box(word_dict['box'], self.image_width, self.image_height),
                                       word_dict['text'],
                                       word_label
                                       ))

    @staticmethod
    def normalize_box(box: List[int], image_width, image_height: int) -> List[int]:
        return [
            int(1000 * (box[0] / image_width)),
            int(1000 * (box[1] / image_height)),
            int(1000 * (box[2] / image_width)),
            int(1000 * (box[3] / image_height)),
        ]

    @staticmethod
    def get_label_prefix(word_index, num_words):
        if num_words == 1:
            return "S-"
        elif word_index + 1 == 1:
            return "B-"
        elif word_index + 1 == num_words:
            return "E-"
        elif 0 < word_index + 1 < num_words:
            return "I-"
        return "O-"


@dataclass
class DocumentText:
    annotations: Dict[str, Any]
    image_height: int
    image_width: int
    text_objects: List[TextObject] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:

        for text_dict in self.annotations['form']:
            if text_dict['text'] != '':
                self.text_objects.append(TextObject(text_dict['box'],
                                                    text_dict['text'],
                                                    text_dict['label'],
                                                    text_dict['words'],
                                                    text_dict['id'],
                                                    self.image_height,
                                                    self.image_width))


@dataclass
class DocumentImage:
    # image: Image
    width: int
    height: int
    image_path: str

    def load_image(self):
        return Image.open(self.image_path).convert("RGB")


@dataclass
class Document:
    image: Image
    annotations: Dict[str, Any]
    image_path: Union[str, None] = None
    annot_path: Union[str, None] = None

    document_image: DocumentImage = field(init=False)
    document_text: DocumentText = field(init=False)

    def __post_init__(self):
        self.document_image = DocumentImage(self.image.width, self.image.height,
                                            self.image_path)  # TODO check whether it is a good idea to load images (?)
        self.document_text = DocumentText(self.annotations, self.document_image.height, self.document_image.width)

    def get_data(self) -> pd.DataFrame:
        data_list = []
        for text_object in self.document_text.text_objects:
            for word_object in text_object.words:
                data_list.append(asdict(word_object))
        return pd.DataFrame(data_list).to_dict(orient="list")


@dataclass
class DocumentWarehouse:
    annotations_list: List[Dict[str, Any]] = field(default_factory=list)
    images_list: List[Image.Image] = field(default_factory=list)
    documents: List[Document] = field(init=False, default_factory=list)

    def __post_init__(self):
        for image, annotations in zip(self.images_list, self.annotations_list):
            self.documents.append(Document(image, annotations))

    def get_all_data(self) -> pd.DataFrame:
        data_list = []
        for document_object in self.documents:
            data_dict = document_object.get_data()
            data_dict['annotation_path'] = document_object.annot_path
            data_dict['image_path'] = document_object.image_path
            data_dict['image'] = document_object.image
            data_list.append(data_dict)
        return pd.DataFrame(data_list)


@dataclass
class TrainingDocumentWarehouse(DocumentWarehouse):
    image_dir: str = field(default_factory=str)
    annotation_dir: str = field(default_factory=str)
    documents: List[Document] = field(init=False, default_factory=list)
    image_file_paths: List[str] = field(init=False, default_factory=list)
    annotations_file_paths: List[str] = field(init=False, default_factory=list)

    def load_data(self) -> None:
        image_files = list(Path(self.image_dir).rglob("*"))  # TODO make here Enum class with possible image extensions
        annotations_files = list(Path(self.annotation_dir).rglob("*.json"))
        self.image_file_paths, self.annotations_file_paths = self.pair_up_files(image_files, annotations_files)
        for (image_file_path, annotations_file_path) in zip(self.image_file_paths, self.annotations_file_paths):
            self.images_list.append(Image.open(image_file_path).convert("RGB"))
            with open(annotations_file_path, 'r') as fp:
                annotations = json.load(fp)
            self.annotations_list.append(annotations)

    def __post_init__(self) -> None:
        self.load_data()
        for (image, annotations, image_file_path, annotations_file_path) in zip(self.images_list,
                                                                                self.annotations_list,
                                                                                self.image_file_paths,
                                                                                self.annotations_file_paths):
            self.documents.append(Document(image, annotations, image_file_path, annotations_file_path))

    @staticmethod
    def get_file_name(path: Path) -> str:
        return str(path).split(os.path.sep)[-1].split(".")[0]

    def pair_up_files(self, image_file_paths, annotations_file_paths):
        image_files = []
        annotation_files = []
        for image_file_path in sorted(image_file_paths, key=self.get_file_name):
            for annotations_file_path in sorted(annotations_file_paths, key=self.get_file_name):
                if self.get_file_name(image_file_path) == self.get_file_name(annotations_file_path):
                    image_files.append(str(image_file_path))
                    annotation_files.append(str(annotations_file_path))
        return image_files, annotation_files

    @staticmethod
    def validate_numpy_arr(arr: np.ndarray) -> Union[None, np.ndarray]:
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def select_data(dataframe: pd.DataFrame, indices: Union[np.ndarray, None]) -> Union[pd.DataFrame, None]:
        if indices is None:
            return None
        return dataframe.iloc[indices]

    def split_indices(self, test_percentage: float = 0.0,
                      validation_percentage: float = 0.0):
        num_of_indices = len(self.documents)
        all_indices = np.arange(num_of_indices)
        np.random.shuffle(all_indices)
        test_size = int(num_of_indices * test_percentage)
        validation_size = int(num_of_indices * validation_percentage)
        test_indices = self.validate_numpy_arr(all_indices[:test_size])
        validation_indices = self.validate_numpy_arr(all_indices[test_size:validation_size + test_size])
        training_indices = self.validate_numpy_arr(all_indices[test_size + validation_size:])
        return test_indices, validation_indices, training_indices

    def get_all_data(self) -> pd.DataFrame:
        data_list = []
        for document_object in self.documents:
            data_dict = document_object.get_data()
            data_dict['image_path'] = document_object.image_path
            data_list.append(data_dict)
        return pd.DataFrame(data_list)

    def get_datasets(self, test_percentage: float = 0.0,
                     validation_percentage: float = 0.0) -> Datasets:
        test_indices, validation_indices, train_indices = \
            self.split_indices(test_percentage=test_percentage, validation_percentage=validation_percentage)
        data: pd.DataFrame = self.get_all_data()
        datasets: Datasets = {'train_dataset': self.select_data(data, train_indices),
                              'test_dataset': self.select_data(data, test_indices),
                              'validation_dataset': self.select_data(data, validation_indices)}
        return datasets
