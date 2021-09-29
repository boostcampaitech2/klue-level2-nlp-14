import torch


class DefaultDataCollator:
    
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if max_length == 'single':
            self.max_length = tokenizer.max_len_single_sentence
        elif max_length == 'pair':
            self.max_length = tokenizer.max_len_sentences_pair
        elif type(max_length) == int:
            self.max_length = max_length
    
    def __call__(self, batch):
        input_ids = [x["input_ids"] for x in batch]
        batch_encoding = self.tokenizer.pad(
            {"input_ids": input_ids},
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            labels = [x["label"] for x in batch]
            batch_encoding.update({"labels": torch.LongTensor(labels)})
        return batch_encoding