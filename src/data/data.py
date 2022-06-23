import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv2Processor
from typing import Dict
from PIL import Image
import pandas as pd


class NERDataset(Dataset):
    def __init__(self, dataset_df: pd.DataFrame, processor: LayoutLMv2Processor, label2id: Dict[str, int]) -> None:
        self.boxes = dataset_df['normalized_box'].to_numpy()
        self.original_boxes = dataset_df['box'].to_numpy()
        self.words = dataset_df['text'].to_numpy()
        self.image_file_names = dataset_df['image_path'].to_numpy()
        self.word_labels = dataset_df['label'].to_numpy()
        self.processor = processor
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.image_file_names)

    def __getitem__(self, idx):
        image = Image.open(self.image_file_names[idx]).convert("RGB")
        words = self.words[idx]
        boxes = self.boxes[idx]
        word_labels = [self.label2id[word_label] for word_label in self.word_labels[idx]]
        encoded_inputs = self.processor(image, words, boxes=boxes, word_labels=word_labels,
                                        return_special_tokens_mask=True,
                                        return_offsets_mapping=True,
                                        padding="max_length", truncation=True, return_tensors='pt')
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze()
        encoded_inputs.update({"original_bbox": torch.Tensor(self.original_boxes[idx])})
        return encoded_inputs
