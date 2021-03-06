import hydra
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.data import TrainingDocumentWarehouse, NERDataset
from src.model import ModelTrainer


@hydra.main(config_path='conf', config_name='config', version_base="1.2")
def main(cfg) -> None:
    cfg = instantiate(cfg)
    device = torch.device(cfg.training.device)

    document_warehouse = TrainingDocumentWarehouse(cfg.dataset.images_dir,
                                                   cfg.dataset.annotations_dir)
    dataset = document_warehouse.get_datasets(test_percentage=cfg.training.test_percentage,
                                              validation_percentage=cfg.training.validation_percentage)

    train_dataset = NERDataset(dataset['train_dataset'], cfg.model.processor, cfg.dataset.label2id)
    test_dataset = NERDataset(dataset['test_dataset'], cfg.model.processor, cfg.dataset.label2id)
    # validation_dataset = NERDataset(dataset['validation_dataset'], cfg.model.processor, cfg.dataset.label2id)

    training_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    validation_dataloader = None  # DataLoader(validation_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.training.batch_size)

    trainer = ModelTrainer(model=cfg.model.model.to(device),
                           device=device,
                           training_dataloader=training_dataloader,
                           validation_dataloader=validation_dataloader,
                           test_dataloader=test_dataloader,
                           epochs=cfg.training.epoch_count,
                           use_wandb=cfg.params.use_wandb,
                           project_name=cfg.params.project_name,
                           labels=list(cfg.dataset.label2id.keys()))
    trainer.run_trainer()


if __name__ == "__main__":
    main()
