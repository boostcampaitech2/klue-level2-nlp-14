from transformers import Trainer
import datasets
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler # pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
from ..utils import (
    LOSS_MAP,
)

class DefaultTrainer(Trainer):
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