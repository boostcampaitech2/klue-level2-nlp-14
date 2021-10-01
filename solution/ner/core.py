import json
from typing import List, Tuple
from kss import split_sentences
import numpy as np
from .bpetokenizer import KoBpeTokenizer
from .modeling_roberta import RobertaForCharNER


def flatten(texts: List[List[str]]) -> List[str]:
    return [sentence for sentences in texts for sentence in sentences]


"""
TODO
- 영일중 sentence splitter 코드
- 영일중 ner 모델 포팅 및 적용
"""


class NERInterface:
    
    @classmethod
    def from_pretrained(cls, **filenames):
        wsd = filenames.pop("wsd", None)
        wsd_dict = None
        if wsd is not None:
            wsd_dict = json.load(open(wsd, "r"))
        vocab_file = filenames.pop("vocab", None)
        vocab = None
        if vocab_file is not None:
            tokenizer = KoBpeTokenizer.from_file(vocab_file)
        label_file = filenames.pop("label", None)
        label = None
        if label_file is not None:
            label = json.load(open(label_file, "r"))
        model_file = filenames.pop("model", None)
        if model_file is not None:
            model = RobertaForCharNER.from_pretrained(
                model_file
            )
        return NERInterface(
            model=model,
            charbpe=tokenizer,
            label=label,
            wsd_dict=wsd_dict,
        )
    
    def __init__(
        self,
        model,
        charbpe,
        label,
        wsd_dict,
    ):
        self._model = model
        self._sent_tokenizer = split_sentences
        self.bpe = charbpe
        self._label = label
        self._tags = {
            "PS": "PERSON",
            "LC": "LOCATION",
            "OG": "ORGANIZATION",
            "AF": "ARTIFACT",
            "DT": "DATE",
            "TI": "TIME",
            "CV": "CIVILIZATION",
            "AM": "ANIMAL",
            "PT": "PLANT",
            "QT": "QUANTITY",
            "FD": "STUDY_FIELD",
            "TR": "THEORY",
            "EV": "EVENT",
            "MT": "MATERIAL",
            "TM": "TERM",
        }
        self._wsd_dict = wsd_dict
        self._wsd = None
        self._cls2cat = None
        self._quant2cat = None
        self._term2cat = None
        
    @property
    def id2label(self):
        return {i: label for label, i in self._label.items()}
    
    @property
    def label2id(self):
        return {label: i for label, i in self._label.items()}
        
    def apply_dict(self, tags: List[Tuple[str, str]]):
        """
        Apply pre-defined dictionary to get detail tag info
        Args:
            tags (List[Tuple[str, str]]): inference word-tag pair result
        Returns:
            List[Tuple[str, str]]: dict-applied result
        """
        result = []
        for pair in tags:
            word, tag = pair
            if (tag in self._wsd_dict.keys()) and (word in self._wsd_dict[tag]):
                result.append((word, self._wsd_dict[tag][word].upper()))
            else:
                result.append(pair)
        return result
    
    def _apply_wsd(self, tags: List[Tuple[str, str]]):
        """
        Apply Word Sense Disambiguation to get detail tag info
        Args:
            tags (List[Tuple[str, str]]): inference word-tag pair result
        Returns:
            List[Tuple[str, str]]: wsd-applied result
        """
        # https://github.com/kakaobrain/pororo/blob/master/pororo/tasks/named_entity_recognition.py#L289
        raise NotImplementedError
        
    def _postprocess(self, tags: List[Tuple[str, str]]):
        """
        Postprocess characted tags to concatenate BIO
        Args:
            tags (List[Tuple[str, str]]): characted token and its corresponding tag tuple list
        Returns:
            List(Tuple[str, str]): postprocessed entity token and its corresponding tag tuple list
        """

        def _remove_tail(tag: str):
            if "-" in tag:
                tag = tag[:-2]
            return tag

        result = list()

        tmp_word = tags[0][0]
        prev_ori_tag = tags[0][1]
        prev_tag = _remove_tail(prev_ori_tag)
        for _, pair in enumerate(tags[1:]):
            char = pair[0]
            ori_tag = pair[1]
            tag = _remove_tail(ori_tag)
            if ("▁" in char) and ("-I" not in ori_tag):
                result.append((tmp_word, prev_tag))
                result.append((" ", "O"))

                tmp_word = char
                prev_tag = tag
                continue

            if (tag == prev_tag) and (("-I" in ori_tag) or "O" in ori_tag):
                tmp_word += char
            elif (tag != prev_tag) and ("-I" in ori_tag) and (tag != "O"):
                tmp_word += char
            else:
                result.append((tmp_word, prev_tag))
                tmp_word = char

            prev_tag = tag
        result.append((tmp_word, prev_tag))

        result = [(pair[0].replace("▁", " ").strip(),
                   pair[1]) if pair[0] != " " else (" ", "O")
                  for pair in result]
        return result
    
    def sent_tokenize(self, texts: List[str]) -> List[List[str]]:
        texts = [text.strip() for text in texts]
        texts = self._sent_tokenizer(texts)
        num_sentences = [len(sentences) for sentences in texts]
        return texts, num_sentences
    
    def __call__(self, texts: str, **kwargs):
        """
        Conduct named entity recognition with character BERT
        Args:
            text: (str) sentence to be sequence labeled
            apply_wsd: (bool) whether to apply wsd to get more specific label information
            ignore_labels: (list) labels to be ignored
        Returns:
            List[Tuple[str, str]]: token and its predicted tag tuple list
        """
        apply_wsd = kwargs.get("apply_wsd", False)
        ignore_labels = kwargs.get("ignore_labels", [])
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Sentence split
        texts, n_sents = self.sent_tokenize(texts)
        texts = flatten(texts)
        
        # tokenize
        tokens = self.bpe(texts, return_tokens=True)
        input_ids = self.bpe(
            texts, 
            add_special_tokens=True, 
            return_tensors=True
        )
        # predict tags
        logits = self._model(input_ids).logits
        results = logits[:, 1:-1:, :].argmax(dim=-1).cpu().numpy()
        
        labelmap = lambda x: self.id2label[x + self.bpe.nspecial]
        labels = np.vectorize(labelmap)(results)
        token_label_pairs = [
            [
                (tk, l)
                for tk, l in zip(sent, label)
            ] for sent, label in zip(tokens, labels)
        ]
        # Post processing
        results = []
        ix = 0
        for batch_ix in range(len(n_sents)):
            n_sent = n_sents[batch_ix]
            result = []
            for _ in range(n_sent):
                res = []
                sentence = token_label_pairs[ix]
                for pair in self._postprocess(sentence):
                    if pair[1] not in ignore_labels:
                        r = (pair[0], self._tags.get(pair[1], pair[1]))
                        if apply_wsd:
                            r = self._apply_wsd(r)
                        res.append(r)
                result.extend(self.apply_dict(res))
                result.extend([(" ", "O")])
                ix += 1
            results.append(result[:-1])
            
        return results