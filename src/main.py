import hydra
from hydra.core.config_store import ConfigStore
from config import NERConfig
from typing import Optional
from data import DocumentWarehouse, NERDataset
from model import ModelTrainer
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from torch.utils.data import DataLoader
import torch

cs = ConfigStore.instance()
cs.store(name='ner_config', node=NERConfig)


@hydra.main(config_path='conf', config_name='config', version_base="1.2")
def main(cfg: Optional[NERConfig]):
    device = torch.device('cuda')
    processor = LayoutLMv3Processor.from_pretrained(cfg.model.name,
                                                    apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(cfg.model.name,
                                                             num_labels=len(cfg.dataset.label2id)).to(device)
    dataset = DocumentWarehouse(cfg.dataset.images_dir,
                                cfg.dataset.annotations_dir).get_datasets(test_percentage=0.15, validation_percentage=0.15)
    train_dataset = NERDataset(dataset['train_dataset'], processor, cfg.dataset.label2id)
    test_dataset = NERDataset(dataset['test_dataset'], processor, cfg.dataset.label2id)
    validation_dataset = NERDataset(dataset['validation_dataset'], processor, cfg.dataset.label2id)

    training_dataloader = DataLoader(train_dataset, batch_size=cfg.params.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.params.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.params.batch_size)
    trainer = ModelTrainer(model=model,
                           device=device,
                           training_dataloader=training_dataloader,
                           validation_dataloader=validation_dataloader,
                           test_dataloader=test_dataloader,
                           epochs=cfg.params.epoch_count,
                           use_wandb=True,
                           project_name="NERv3",
                           labels=list(cfg.dataset.label2id.keys()))
    trainer.run_trainer()


if __name__ == "__main__":
    main()
