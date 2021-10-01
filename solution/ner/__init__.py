from .core import NERInterface
from .bpetokenizer import KoBpeTokenizer
from .modeling_roberta import RobertaForCharNER


SUPPORTED_LANGS = ["ko", "en", "zh", "ja"]

NER_FILES = {
    "ko": dict(
        label="ner/label.json",
        vocab="ner/vocab.json",
        wsd="ner/wsd.json",
        model="jinmang2/roberta-ko-ner"
    )
}