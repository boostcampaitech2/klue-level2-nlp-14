import torch
from .mlm import mask_tokens


class DefaultDataCollator:
    """ Default Data Collator
    Attributes:
        tokenizer:  Tokenizer for text tokenization.
        max_length: The maximum length of the sequence.
    """
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
    
    
class MLMDataCollator:
    """ Data Collator for MLM(Masked Language Model)
    Attributes:
        tokenizer:  Tokenizer for text tokenization.
        max_length: The maximum length of the sequence.
        mlm_prob:   The ratio of mask tokens.
    """
    def __init__(self, tokenizer, max_length=512, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_prob = mlm_prob
        
    def __call__(self, batch):
        # batch_encoding = self.tokenizer(
        #     text=[x["sentence"] for x in batch], 
        #     return_tensors="pt",
        #     padding=True,
        #     max_length=self.max_length,
        #     truncation=True,
        # )
        input_ids = [x["input_ids"] for x in batch]
        batch_encoding = self.tokenizer.pad(
            {"input_ids": input_ids},
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs, labels = mask_tokens(
            batch_encoding["input_ids"],
            self.tokenizer,
            self.mlm_prob,
        )
        return {"input_ids": inputs, "labels": labels}