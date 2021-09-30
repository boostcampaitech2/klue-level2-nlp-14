from sadice import SelfAdjDiceLoss
from torch.utils.data import DataLoader
from transformers import Trainer
from torchsampler import ImbalancedDatasetSampler

class ImbalancedSamplerTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset

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
        logits = outputs.logits

        criterion = SelfAdjDiceLoss()
        loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss