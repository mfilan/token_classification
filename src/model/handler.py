from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor


class ModelHandler:
    def __init__(self, pretrained_model_name: str, num_of_labels: int):
        if "layoutlmv3" in pretrained_model_name:
            self.processor = LayoutLMv3Processor.from_pretrained(pretrained_model_name,
                                                                 apply_ocr=False)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(pretrained_model_name,
                                                                          num_labels=num_of_labels)
        else:
            raise NotImplementedError(f"Logic for {pretrained_model_name} is not implemented!")
