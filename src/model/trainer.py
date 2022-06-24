import logging
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm.auto import tqdm, trange

logger = logging.getLogger("NER")


class ModelTrainer:
    """
    Class responsible for training model

    Parameters
    ----------
        model (torch.nn.Module): instance of pytorch Module representing model to train
        device (torch.device): device to train on cuda/cpu
        optimizer (torch.optim.Optimizer): optimizer used for training
        training_DataLoader (torch.utils.data.Dataset): training dataset
        validation_dataLoader (torch.utils.data.Dataset): validation dataset
        test_dataLoader (torch.utils.data.Dataset): test dataset
        lr_scheduler (torch.optim.lr_scheduler): scheduler used in training
        epochs (int): number of epochs
        epoch (int): starting epoch
        use_wandb (bool): whether to use wandb for logging
        project_name (str): project name used for wandb logging
        save_model_name (str): file name for trained model

    """

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 training_dataloader: torch.utils.data.Dataset,
                 validation_dataloader: torch.utils.data.Dataset = None,
                 test_dataloader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 10,
                 epoch: int = 0,
                 use_wandb: bool = False,
                 project_name: str = "NERv2",
                 save_model_name: str = join("models", "model.pth"),
                 labels: list = None
                 ):

        logger.info('Initializing model trainer...')
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-05)
        self.lr_scheduler = lr_scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.use_wandb = use_wandb
        self.save_model_name = Path(save_model_name)
        self.save_model_name.parent.mkdir(parents=True, exist_ok=True)
        self.labels = labels
        self.index_to_label = {i: label for i, label in enumerate(self.labels)}

        if self.use_wandb:
            wandb.init(project=project_name, config={'initial_lr': self.optimizer.defaults['lr'],
                                                     'epochs': self.epochs})
            wandb.define_metric("validation_loss", step_metric="step")

        self.training_step = 0
        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def train_log(self, loss):
        """
        Function responsible for logging training loss to wandb

        Parameters
        ----------
            loss (float): running training loss

        """
        wandb.log({"epoch": self.epoch, "loss": loss}, step=self.training_step)

    def validation_log(self, loss):
        """
        Function responsible for logging validation loss to wandb

        Parameters
        ----------
            loss (float): running validation loss

        """
        wandb.log({"epoch": self.epoch, "validation_loss": loss}, step=self.training_step)

    def save_labels(self):
        with open(join(str(self.save_model_name.parent), "labels.txt"), 'w') as fp:
            fp.write("\n".join(sorted(list(self.labels))))

    def run_trainer(self):
        """
        Wrapper of _run_trainer responsible for error handling
        """
        logger.info('Training has started...')

        try:
            train_output = self._run_trainer()
            if self.use_wandb:
                wandb.finish()
            logger.info(f"Model saved as {str(self.save_model_name)}")
            torch.save(self.model, self.save_model_name)
            self.save_labels()
            return train_output
        except KeyboardInterrupt:
            if self.use_wandb:
                wandb.finish()
            logger.warning("KeyboardInterrupt detected! Saving model...")
            torch.save(self.model, str(self.save_model_name))
            self.save_labels()

    def _run_trainer(self):
        """
        Function responsible for running _train() and _validate
        """

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_dataloader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # learning rate scheduler step
        if self.test_dataloader is not None:
            classification_report = pd.DataFrame(self._test()).T
            if self.use_wandb:
                wandb.log({'classification report': wandb.Table(dataframe=classification_report.reset_index())})
                wandb.finish()
            logger.info(classification_report.to_string())
            results_name = f'{str(self.save_model_name.parent)}/' \
                           f'{str(self.save_model_name.name).rstrip(".pth")}_classification_report.csv'
            classification_report.to_csv(f'{results_name}')
            logger.info(f"Results saved as {results_name}")
            return {'training_loss': self.training_loss,
                    'validation_loss': self.validation_loss,
                    'learning_rate': self.learning_rate,
                    'classification_report': classification_report}

        if self.use_wandb:
            wandb.finish()

        return {'training_loss': self.training_loss,
                'validation_loss': self.validation_loss,
                'learning_rate': self.learning_rate}

    def _train(self):
        """
        Function responsible for training model
        """
        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_dataloader), 'Training', total=len(self.training_dataloader),
                          leave=False)

        for i, batch in batch_iter:
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input_ids=batch['input_ids'].to(self.device),
                             attention_mask=batch['attention_mask'].to(self.device),
                             bbox=batch['bbox'].to(self.device),
                             labels=batch['labels'].to(self.device),
                             pixel_values=batch['pixel_values'].to(self.device))  # one forward pass
            loss = out.loss  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            if self.use_wandb:
                self.train_log(loss_value)
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
            self.training_step += 1
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):
        """
        Function responsible for validating model
        """
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_dataloader), 'Validation', total=len(self.validation_dataloader),
                          leave=False)

        for i, batch in batch_iter:
            with torch.no_grad():
                out = self.model(input_ids=batch['input_ids'].to(self.device),
                                 attention_mask=batch['attention_mask'].to(self.device),
                                 bbox=batch['bbox'].to(self.device),
                                 labels=batch['labels'].to(self.device),
                                 pixel_values=batch['pixel_values'].to(self.device))  # one forward pass
                loss = out.loss
                loss_value = loss.item()
                valid_losses.append(loss_value)
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        mean_losses = np.mean(valid_losses)
        if self.use_wandb:
            self.validation_log(mean_losses)

        self.validation_loss.append(mean_losses)

        batch_iter.close()

    @staticmethod
    def get_labels(path):
        """
        Reads labels from file

        Parameters
        ----------
            path (str): path to file containing labels

        Returns
        -------
            list containing labels
        """
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels

    def _test(self):

        from datasets import load_metric
        metric = load_metric("seqeval")

        # put model in evaluation mode
        self.model.eval()
        for batch in tqdm(self.test_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = self.model(input_ids=batch['input_ids'].to(self.device),
                                     attention_mask=batch['attention_mask'].to(self.device),
                                     bbox=batch['bbox'].to(self.device),
                                     labels=batch['labels'].to(self.device),
                                     pixel_values=batch['pixel_values'].to(self.device))  # one forward pass

                # predictions
                predictions = outputs.logits.argmax(dim=2)

                # Remove ignored index (special tokens)
                true_predictions = [
                    [self.index_to_label[int(p.detach().cpu().numpy())] for (p, l) in zip(prediction, label) if
                     l != -100]
                    for prediction, label in zip(predictions, batch['labels'])
                ]

                true_labels = [
                    [self.index_to_label[int(l.detach().cpu().numpy())] for (p, l) in zip(prediction, label) if
                     l != -100]
                    for prediction, label in zip(predictions, batch['labels'])
                ]

                metric.add_batch(predictions=true_predictions, references=true_labels)

        final_score = metric.compute()
        return final_score
