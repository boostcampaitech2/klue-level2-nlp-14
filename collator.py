import torch


class DataCollator:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        batch_encoding = self.tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
        )
        if "label" in batch[0]:
            labels = [x["label"] for x in batch]
            batch_encoding.update({"labels": torch.LongTensor(labels)})
        return batch_encoding