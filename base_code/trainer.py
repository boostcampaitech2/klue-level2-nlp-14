import torch
from loss import FocalLoss
from torch.utils.data import DataLoader
from transformers import Trainer
from torchsampler import ImbalancedDatasetSampler
from loss import FocalLoss, DiceLoss, CrossEntropyClassWeight
from collections import defaultdict

class DefaultTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        criterion = FocalLoss(gamma=0.5)
        loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss