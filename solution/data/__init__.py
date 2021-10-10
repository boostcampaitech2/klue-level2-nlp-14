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
    """
    KLUE Baseline preprocessing function. Follow the steps bellow.
    1. mark entity spans
        - "<obj>조지 해리슨</obj>이 쓰고 <subj>비틀즈</subj>가"
    2. convert example to features
        - fix tokenization errors
        - convert tokens to input_ids
    3. remove columns
        - except `input_ids`, `labels`

    Args:
        dataset: huggingface Dataset object used for training
        tokenizer: Tokenizer object (maybe huggingface/transformers' PreTrainedTokenizerFast)
        task_infos: Task information. e.g., entity marker, number of classes, etc.
        mode: Whether this function is used in train or not

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
    """
    KLUE Baseline preprocessing function with entity type. Follow the steps bellow.
    1. type mark entity spans
        - "<obj:PER>조지 해리슨</obj:PER>이 쓰고 <subj:ORG>비틀즈</subj:ORG>가"
    2. convert example to features
        - fix tokenization errors
        - convert tokens to input_ids
    3. remove columns
        - except `input_ids`, `labels`

    Args:
        dataset: huggingface Dataset object used for training
        tokenizer: Tokenizer object (maybe huggingface/transformers' PreTrainedTokenizerFast)
        task_infos: Task information. e.g., entity marker, number of classes, etc.
        mode: Whether this function is used in train or not
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
    """
    KLUE Baseline preprocessing function with entity embedding. Follow the steps bellow.
    1. mark entity spans
        - <subj>subject entity word</subj> ... <obj>object entity word</obj>
    2. convert example to features
        - fix tokenization errors
        - convert tokens to input_ids
    3. get entity ids
        - if input tokens is ["<subj>", "이순신", "</subj>", "##은", "<obj>", "조선", "##중기", "</obj>", "##의", "무신", "##이다"],
          then entity ids is [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    4. remove columns
        - except `input_ids`, `entity_ids`, `labels`

    Args:
        dataset: huggingface Dataset object used for training
        tokenizer: Tokenizer object (maybe huggingface/transformers' PreTrainedTokenizerFast)
        task_infos: Task information. e.g., entity marker, number of classes, etc.
        mode: Whether this function is used in train or not
    """

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
    """
    Preprocessing function for recent model. Follow the steps bellow.
    1. mark entity spans
        - <subj>subject entity word</subj> ... <obj>object entity word</obj>
    2. convert example to features
        - fix tokenization errors
        - convert tokens to input_ids
    3. label shaping
        - since recent's label is different from others, shaping is required.
    4. remove columns
        - except `input_ids`, `head_idx`, `labels`
        - `head_idx` is an argument that determines which head result is recieved in RECENT.

    Args:
        dataset: huggingface Dataset object used for training
        tokenizer: Tokenizer object (maybe huggingface/transformers' PreTrainedTokenizerFast)
        task_infos: Task information. e.g., entity marker, number of classes, etc.
        mode: Whether this function is used in train or not
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
