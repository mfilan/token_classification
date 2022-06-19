from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import pandas as pd
from PIL import Image
import os


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
    image_path: str
    annot_path: str
    image: DocumentImage = field(init=False)
    text: DocumentText = field(init=False)

    def __post_init__(self):
        self.load_data(self.image_path, self.annot_path)

    def load_image(self, path: str) -> None:
        image = Image.open(path).convert("RGB")
        self.image = DocumentImage(image.width, image.height,
                                   path)  # TODO check whether it is a good idea to load images (?)

    def load_annotations(self, path: str) -> None:
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        self.text = DocumentText(annotations, self.image.height, self.image.width)

    def load_data(self, image_path: str, annot_path: str) -> None:
        self.load_image(image_path)
        self.load_annotations(annot_path)

    def get_data(self) -> pd.DataFrame:
        data_dict = []
        for text_object in self.text.text_objects:
            for word_object in text_object.words:
                data_dict.append(asdict(word_object))
        return pd.DataFrame(data_dict)

@dataclass
class DocumentWarehouse:
    image_dir: str
    annotation_dir: str
    documents: List[Document] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        image_files = list(Path(self.image_dir).rglob("*")) # TODO make here Enum class with possible image extensions
        annotations_files = list(Path(self.annotation_dir).rglob("*.json"))
        file_pairs = self.pair_up_files(image_files, annotations_files)
        for (image_file_path, annotations_file_path) in file_pairs:
            self.documents.append(Document(image_file_path,annotations_file_path))

    @staticmethod
    def pair_up_files(image_file_paths,annotations_file_paths):

        get_file_name = lambda x: str(x).split(os.path.sep)[-1].split(".")[0]
        files = []
        for image_file_path in sorted(image_file_paths,key = get_file_name):
            for annotations_file_path in sorted(annotations_file_paths,key = get_file_name):
                if get_file_name(image_file_path) == get_file_name(annotations_file_path):
                    files.append((str(image_file_path), str(annotations_file_path)))
        return files


    def get_data(self):
        data_dict = []
        for document_object in self.documents:
            data_dict += document_object.get_data()
        return data_dict