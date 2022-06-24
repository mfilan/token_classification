import re
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
from transformers import LayoutLMv3Processor

import src.utils as utils
from src.data import DocumentWarehouse


class PostProcessor:
    def __init__(self, processor, id2label):
        self.processor = processor
        self.id2label = id2label

    def process(self, model_outputs):
        list_of_entities = []
        for per_document_model_outputs in self.model_outputs_generator(model_outputs):
            scores = self.get_scores(per_document_model_outputs['logits'])
            aggregated_pre_entities = self.aggregate_tokens(scores,
                                                            per_document_model_outputs['offset_mapping'],
                                                            per_document_model_outputs['original_bbox'],
                                                            per_document_model_outputs['input_ids'],
                                                            per_document_model_outputs['special_tokens_mask'])
            filtered_pre_entities = self.filter_pre_entities(aggregated_pre_entities)
            entities = self.aggregate_by_label(filtered_pre_entities)
            list_of_entities.append(entities)
        return list_of_entities

    @staticmethod
    def model_outputs_generator(model_outputs):
        assert len(model_outputs['logits'].shape) == 3
        num_of_documents = model_outputs['logits'].shape[0]
        for document_idx in range(num_of_documents):
            logits = utils.to_numpy(model_outputs["logits"][document_idx])
            original_boxes = utils.to_numpy(model_outputs["original_bbox"][document_idx])
            input_ids = utils.to_numpy(model_outputs["input_ids"][document_idx])
            offset_mapping = utils.to_numpy(model_outputs["offset_mapping"][document_idx])
            special_tokens_mask = utils.to_numpy(model_outputs["special_tokens_mask"][document_idx])
            yield {'logits': logits,
                   'original_bbox': original_boxes,
                   'input_ids': input_ids,
                   'offset_mapping': offset_mapping,
                   'special_tokens_mask': special_tokens_mask}

    def aggregate_tokens(self, scores, offset_mapping, original_boxes, input_ids, special_tokens_mask):
        pre_entities = []
        previous_ind = (0, 0)
        previous_token = ''
        scores_temp = []
        labels_temp = []
        counter = 0
        for idx, token_scores in enumerate(scores):
            if special_tokens_mask[idx]:
                continue
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                if not isinstance(start_ind, int):
                    start_ind = start_ind.item()
                    end_ind = end_ind.item()
                word = self.processor.tokenizer.convert_ids_to_tokens(int(input_ids[idx])).lstrip("Ġ")
                label = self.id2label[int(token_scores.argmax())]
                confidence = token_scores[token_scores.argmax()]
                if start_ind == previous_ind[1]:
                    previous_ind = start_ind, end_ind
                    previous_token += word.strip('#')

                    labels_temp.append(label)
                    scores_temp.append(confidence)
                else:
                    entity = {'text': previous_token, 'scores': np.array(scores_temp), 'labels': labels_temp,
                              'coordinates': original_boxes[counter]}
                    pre_entities.append(entity)
                    previous_ind = start_ind, end_ind
                    previous_token = word
                    labels_temp = [label]
                    scores_temp = [confidence]
                    counter += 1
        return pre_entities

    @staticmethod
    def filter_pre_entities(pre_entities):
        filtered_entities = []
        for pre_entity in pre_entities:
            if set(pre_entity['labels']) != {'Other'}:  # TODO check whether it shouldn't be dynamic
                entity = {'Top': pre_entity['coordinates'][1],
                          'Bottom': pre_entity['coordinates'][3],
                          'Left': pre_entity['coordinates'][0],
                          'Right': pre_entity['coordinates'][2],
                          'Text': pre_entity['text'],
                          'Confidence': round(float(np.nanmean(pre_entity['scores'])), 3),
                          'Label': pre_entity['labels'][0]}
                filtered_entities.append(entity)
        return filtered_entities

    @staticmethod
    def get_scores(logits):
        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
        return scores

    @staticmethod
    def strip_label(label) -> str:
        return re.sub('[BIES]-', '', label)

    def strip_labels(self, labels) -> List[str]:
        return [self.strip_label(label) for label in labels]

    @staticmethod
    def group_entities_by_label(words_blocks, labels) -> List[List[Dict[str, Any]]]:
        grouped_blocks = []
        item = []
        for block in words_blocks:
            for label in labels:
                if block["Label"] == f"B-{label}":
                    item = [block]
                elif block["Label"] == f"I-{label}":
                    item.append(block)
                elif block["Label"] == f"E-{label}":
                    item.append(block)
                    grouped_blocks.append(item)
                    item = []
                elif block["Label"] == f"S-{label}":
                    grouped_blocks.append([block])
                    item = []
        return grouped_blocks

    def join_block_groups(self, grouped_blocks):
        return [self.merge_group(group) for group in grouped_blocks]

    def merge_group(self, group) -> Dict[str, Any]:
        text = ''
        coordinates = []
        confidences = []

        for word_block in group:
            coordinates.append(self.get_box_coordinates(word_block))
            text += word_block['Text'] + ' '
            confidences.append(word_block['Confidence'])
        coordinates = self.merge_coordinates(coordinates)
        text = text.strip(" ")
        entity = {'Top': coordinates[1],
                  'Bottom': coordinates[3],
                  'Left': coordinates[0],
                  'Right': coordinates[2],
                  'Text': text,
                  'Confidence': round(sum(confidences) / len(confidences), 2),
                  'Label': self.strip_label(group[-1]['Label'])}
        return entity

    @staticmethod
    def get_box_coordinates(word_block) -> List[int]:
        return [word_block["Left"], word_block["Top"], word_block["Right"], word_block["Bottom"]]

    @staticmethod
    def merge_coordinates(coordinates) -> List[float]:
        coordinates = np.array(coordinates)
        coordinates = np.array(coordinates)
        left = np.min(coordinates[:, 0])
        right = np.max(coordinates[:, 2])
        top = np.min(coordinates[:, 1])
        bottom = np.max(coordinates[:, 3])
        return [float(left), float(top), float(right), float(bottom)]

    def aggregate_by_label(self, entities):
        grouped = self.group_entities_by_label(entities, set(self.strip_labels(list(self.id2label.values()))))
        aggregated = self.join_block_groups(grouped)
        return aggregated


class Preprocessor:
    def __init__(self,
                 processor: LayoutLMv3Processor):
        self.processor = processor

    def process(self, images: List[Image.Image],
                annotations: List[Dict[str, Any]]):
        document_warehouse = DocumentWarehouse(images_list=images, annotations_list=annotations)
        model_inputs = document_warehouse.get_all_data().to_dict(orient="list")
        encoded_inputs = self.processor(model_inputs['image'],
                                        model_inputs['text'],
                                        boxes=model_inputs['normalized_box'],
                                        return_special_tokens_mask=True,
                                        return_offsets_mapping=True,
                                        padding="max_length", truncation=True, return_tensors='pt')
        model_inputs['box'] = np.array([self.pad_boxes(i) for i in model_inputs['box']])
        encoded_inputs.update({"original_bbox": torch.Tensor(model_inputs['box'])})
        return encoded_inputs

    @staticmethod
    def pad_boxes(document_boxes: List[List[int]]) -> np.ndarray:
        document_boxes = np.array(document_boxes)
        length = len(document_boxes)
        padding = int(max(512 - length, 0))
        return np.array(np.pad(document_boxes, ((0, padding), (0, 0)))[:512])


class InferenceModelHandler:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model_path = model_path
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()

    def process(self, model_inputs):
        outputs = dict(self.model(input_ids=model_inputs['input_ids'].to(self.device),
                                  attention_mask=model_inputs['attention_mask'].to(self.device),
                                  bbox=model_inputs['bbox'].to(self.device),
                                  pixel_values=model_inputs['pixel_values'].to(self.device)))
        outputs.update({'offset_mapping': model_inputs['offset_mapping'],
                        'input_ids': model_inputs['input_ids'],
                        'special_tokens_mask': model_inputs['special_tokens_mask'],
                        'original_bbox': model_inputs['original_bbox']})
        return outputs


class InferencePipeline:
    def __init__(self,
                 processor: LayoutLMv3Processor,
                 model_path: str,
                 device: torch.device,
                 id2label):
        self.preprocessor = Preprocessor(processor)
        self.model_inference = InferenceModelHandler(model_path, device)
        self.postprocessor = PostProcessor(processor, id2label)

    def predict(self, images: List[Image.Image],
                annotations: List[Dict[str, Any]]):
        model_inputs = self.preprocessor.process(images, annotations)
        model_outputs = self.model_inference.process(model_inputs)
        processed_output = self.postprocessor.process(model_outputs)
        return processed_output
