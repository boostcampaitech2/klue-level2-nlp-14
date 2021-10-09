from functools import partial

from .collator import (
    DefaultDataCollator,
    MLMDataCollator,
    EntityDataCollator,
)
from .entity_tagging import mark_entity_spans, convert_example_to_features
from .entity_tagging import mark_type_entity_spans, convert_type_example_to_features
from .entity_tagging import get_entity_embedding
from .split import kfold_split


COLLATOR_MAP = {
    "default": DefaultDataCollator,
    "mlm": MLMDataCollator,
    "entity_embedding": EntityDataCollator,
}


def entity_tagging(
    dataset,
    tokenizer=None,
    task_infos=None,
    mode="train",
):
    """ Entity tagging
    Examples:
        "<obj>조지 해리슨</obj>이 쓰고 <subj>비틀즈</subj>가"
    Args:
        dataset:    Original dataset
        tokenizer:  Tokenizer for text tokenization
        task_info:  Project task informations
    Returns:
        Tokenized dataset from which unnecessary columns have been deleted.
    """
    markers = task_infos.markers
    _mark_entity_spans = partial(mark_entity_spans, **markers)
    _convert_example_to_features = partial(
        convert_example_to_features,
        tokenizer=tokenizer,
        **markers,
    )
    examples = dataset.map(_mark_entity_spans)
    tokenized_datasets = examples.map(_convert_example_to_features)

    # remove unused feature names
    column_names = tokenized_datasets.column_names
    if isinstance(column_names, dict):
        column_names = list(column_names.values())[0]
    column_names.pop(column_names.index("input_ids"))
    if mode == "train":
        column_names.pop(column_names.index("label"))

    return tokenized_datasets.remove_columns(column_names)


def type_entity_tagging(
    dataset,
    tokenizer=None,
    task_infos=None,
    mode="train",
):
    """ Type Entity tagging
    Examples:
        "<obj:PER>조지 해리슨</obj:PER>이 쓰고 <subj:ORG>비틀즈</subj:ORG>가"
    Args:
        dataset:    Original dataset
        tokenizer:  Tokenizer for text tokenization
        task_info:  Project task informations
    Returns:
        Tokenized dataset from which unnecessary columns have been deleted.
    """
    _convert_type_example_to_features = partial(
        convert_type_example_to_features,
        tokenizer=tokenizer,
    )
    examples = dataset.map(mark_type_entity_spans)
    tokenized_datasets = examples.map(_convert_type_example_to_features)

    # remove unused feature names
    column_names = tokenized_datasets.column_names
    if isinstance(column_names, dict):
        column_names = list(column_names.values())[0]
    column_names.pop(column_names.index("input_ids"))
    if mode == "train":
        column_names.pop(column_names.index("label"))

    return tokenized_datasets.remove_columns(column_names)


def entity_tagging_embedding(
    dataset,
    tokenizer=None,
    task_info=None,
    mode="train",
):
    markers = task_info.markers
    _mark_entity_spans = partial(mark_entity_spans, **markers)
    _convert_example_to_features = partial(
        convert_example_to_features,
        tokenizer=tokenizer,
        **markers,
    )
    _get_entity_embedding = partial(
        get_entity_embedding,
        tokenizer=tokenizer,
        **markers,
    )
    examples = dataset.map(_mark_entity_spans)
    tokenized_datasets = examples.map(_convert_example_to_features)
    tokenized_datasets = tokenized_datasets.map(_get_entity_embedding)

    # remove unused feature names
    column_names = tokenized_datasets.column_names
    if isinstance(column_names, dict):
        column_names = list(column_names.values())[0]
    column_names.pop(column_names.index("input_ids"))
    column_names.pop(column_names.index("entity_ids"))
    if mode == "train":
        column_names.pop(column_names.index("label"))

    return tokenized_datasets.remove_columns(column_names)


PREPROCESSING_PIPELINE = {
    "entity_tagging": entity_tagging,
    "type_entity_tagging": type_entity_tagging,
    "entity_tagging_embedding": entity_tagging_embedding,
}
