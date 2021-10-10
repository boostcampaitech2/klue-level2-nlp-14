from functools import partial

from .collator import (
    DefaultDataCollator,
    MLMDataCollator,
    RecentDataCollator,
    EntityDataCollator,
)
from .entity_tagging import mark_entity_spans, convert_example_to_features
from .entity_tagging import mark_type_entity_spans, convert_type_example_to_features
from .entity_tagging import get_entity_embedding
from .split import kfold_split


COLLATOR_MAP = {
    "default": DefaultDataCollator,
    "mlm": MLMDataCollator,
    "recent": RecentDataCollator,
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
        mode:       Train mode or another mode.
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
    """ Entity tagging with embedding """
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


def recent_entity_tagging(
    dataset,
    tokenizer=None,
    task_infos=None,
    mode="train",
):
    """ Entity tagging for recent model """
    markers = task_infos.markers
    _mark_entity_spans = partial(mark_entity_spans, **markers)
    _convert_example_to_features = partial(
        convert_example_to_features,
        tokenizer=tokenizer,
        **markers,
    )
    examples = dataset.map(_mark_entity_spans)
    tokenized_datasets = examples.map(_convert_example_to_features)

    # Label making
    def label_shape(example):
        subj_ent_type = example["subject_entity"]["type"]
        obj_ent_type = example["object_entity"]["type"]
        type_pair = f"{subj_ent_type}_{obj_ent_type}"
        head_id = task_infos.type_pair_to_head_id[type_pair]
        label_name = task_infos.id2label[example["label"]]
        # RECENT는 multi head model입니다.
        output = {"head_ids": [head_id],}
        # RECENT 알고리즘의 label은 1 + n_heads입니다
        # 앞의 1개는 실제 label에 대한 index이며
        # [1:]는 head의 개수만큼 label index를 부여합니다
        if mode == "train":
            label = [example["label"]]
            for _, hlabels in task_infos.head_id_to_head_labels.items():
                # TODO: head id가 같은 경우만 학습
                label += [hlabels.get(label_name, 0)]
            output.update({"label": label})
        return output

    tokenized_datasets = tokenized_datasets.map(label_shape)

    # remove unused feature names
    column_names = tokenized_datasets.column_names
    if isinstance(column_names, dict):
        column_names = list(column_names.values())[0]
    column_names.pop(column_names.index("input_ids"))
    column_names.pop(column_names.index("head_ids"))
    if mode == "train":
        column_names.pop(column_names.index("label"))

    return tokenized_datasets.remove_columns(column_names)


PREPROCESSING_PIPELINE = {
    "entity_tagging": entity_tagging,
    "type_entity_tagging": type_entity_tagging,
    "recent_entity_tagging": recent_entity_tagging,
    "entity_tagging_embedding": entity_tagging_embedding,
}
