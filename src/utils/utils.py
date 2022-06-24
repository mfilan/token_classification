import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image
from typing import List, Dict, Any, Union


def to_numpy(vector: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(vector, torch.Tensor):
        return vector.detach().cpu().numpy()
    return vector


def visualize_predictions(image: Image.Image, processed_output: List[Dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    label2color = {'question': 'blue', 'answer': 'green', 'header': 'orange'}
    font = ImageFont.load_default()
    for block in processed_output:
        box = (block['Left'], block['Top'], block['Right'], block['Bottom'])
        draw.rectangle(box, outline=label2color[block['Label']], width=2)
        draw.text((box[0] + 10, box[1] - 10), block['Label'], fill=label2color[block['Label']], font=font)
    return image
