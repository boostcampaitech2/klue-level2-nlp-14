from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)

from .modeling_bert import *
from .modeling_electra import *
from .modeling_roberta import *
from .modeling_mt5 import *
from .modeling_xlm import *

# Get __init__ modules
import sys

mod = sys.modules[__name__]


# Get model
def basic_model_init(model_args, task_infos, tokenizer):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=task_infos.num_labels,
        cache_dir=model_args.model_cache_dir,
        id2label=task_infos.id2label,
        label2id=task_infos.label2id,
    )
    model_cls = getattr(mod, model_args.architectures,
                        AutoModelForSequenceClassification)
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.model_cache_dir,
    )
    if model.config.vocab_size < len(tokenizer):
        print("resize...")
        model.resize_token_embeddings(len(tokenizer))
    return model


def recent_model_init(model_args, task_infos, tokenizer):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=task_infos.num_labels,
        cache_dir=model_args.model_cache_dir,
        id2label=task_infos.id2label,
        label2id=task_infos.label2id,
    )
    config.dense_type = model_args.dense_type
    config.act_type = model_args.act_type
    config.num_labels_per_head = [
        len(label_id) for label_id in task_infos.head_id_to_label_id
    ]
    config.head2label = task_infos.head_id_to_label_id
    model_cls = getattr(mod, model_args.architectures,
                        RobertaForKlueRecent)
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.model_cache_dir,
    )
    if model.config.vocab_size < len(tokenizer):
        print("resize...")
        model.resize_token_embeddings(len(tokenizer))
    return model


MODEL_INIT_FUNC = {
    "basic": basic_model_init,
    "recent": recent_model_init,
}
