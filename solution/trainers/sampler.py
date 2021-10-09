from transformers import Trainer
import datasets
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler # pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
from ..utils import ( 
    LOSS_MAP,
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.args.loss == "weight":
            criterion = LOSS_MAP[self.args.loss](self.train_dataset['label'])
        else:
            criterion = LOSS_MAP[self.args.loss]()
        
        loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

class BalancedSamplerTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        참고 : https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.get_train_dataloader
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")


        def get_label(dataset):
            return dataset["label"]

        train_sampler = ImbalancedDatasetSampler(
            train_dataset, callback_get_label=get_label
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.args.loss == "weight":
            criterion = LOSS_MAP[self.args.loss](self.train_dataset['label'])
        else:
            criterion = LOSS_MAP[self.args.loss]()
        
        loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
