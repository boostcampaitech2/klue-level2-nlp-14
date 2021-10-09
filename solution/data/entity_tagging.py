import torch
import logging
from functools import partial
from typing import Tuple, List, Any, Dict
import transformers
from transformers import BertTokenizerFast, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def mark_entity_spans(examples,
                      subject_start_marker: str, subject_end_marker: str,
                      object_start_marker: str, object_end_marker: str):

    def _mark_entity_spans(
        text: str,
        subject_range=Tuple[int, int],
        object_range=Tuple[int, int]
    ) -> str:
        """ Adds entity markers to the text to identify the subject/object entities.
        Args:
            text: Original sentence
            subject_range: Pair of start and end indices of subject entity
            object_range: Pair of start and end indices of object entity
        Returns:
            A string of text with subject/object entity markers
        """
        if subject_range < object_range:
            segments = [
                text[: subject_range[0]],
                subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                subject_end_marker,
                text[subject_range[1] + 1 : object_range[0]],
                object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                object_end_marker,
                text[object_range[1] + 1 :],
            ]
        elif subject_range > object_range:
            segments = [
                text[: object_range[0]],
                object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                object_end_marker,
                text[object_range[1] + 1 : subject_range[0]],
                subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                subject_end_marker,
                text[subject_range[1] + 1 :],
            ]
        else:
            raise ValueError("Entity boundaries overlap.")

        marked_text = "".join(segments)

        return marked_text

    subject_entity = examples["subject_entity"]
    object_entity = examples["object_entity"]

    text = _mark_entity_spans(
        examples["sentence"],
        (subject_entity["start_idx"], subject_entity["end_idx"]),
        (object_entity["start_idx"], object_entity["end_idx"]),
    )
    return {"text": text}


def mark_type_entity_spans(examples):

    subject_entity = examples["subject_entity"]
    object_entity = examples["object_entity"]

    subj_type = subject_entity["type"]
    obj_type = object_entity["type"]

    subject_start_marker = f"<subj:{subj_type}>"
    subject_end_marker   = f"</subj:{subj_type}>"
    object_start_marker  = f"<obj:{obj_type}>"
    object_end_marker    = f"</obj:{obj_type}>"

    def _mark_entity_spans(
        text: str,
        subject_range=Tuple[int, int],
        object_range=Tuple[int, int],
    ) -> str:
        """ Adds entity markers to the text to identify the subject/object entities.
        Args:
            text: Original sentence
            subject_range: Pair of start and end indices of subject entity
            object_range: Pair of start and end indices of object entity
        Returns:
            A string of text with subject/object entity markers
        """
        if subject_range < object_range:
            segments = [
                text[: subject_range[0]],
                subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                subject_end_marker,
                text[subject_range[1] + 1 : object_range[0]],
                object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                object_end_marker,
                text[object_range[1] + 1 :],
            ]
        elif subject_range > object_range:
            segments = [
                text[: object_range[0]],
                object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                object_end_marker,
                text[object_range[1] + 1 : subject_range[0]],
                subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                subject_end_marker,
                text[subject_range[1] + 1 :],
            ]
        else:
            raise ValueError("Entity boundaries overlap.")

        marked_text = "".join(segments)

        return marked_text

    text = _mark_entity_spans(
        examples["sentence"],
        (subject_entity["start_idx"], subject_entity["end_idx"]),
        (object_entity["start_idx"], object_entity["end_idx"]),
    )
    return {"text": text}


def convert_example_to_features(
    examples,
    tokenizer,
    subject_start_marker: str,
    subject_end_marker: str,
    object_start_marker: str,
    object_end_marker: str
) -> Dict[str, List[Any]]:

    tokenizer_type = check_tokenizer_type(tokenizer)
  
    def fix_tokenization_error(text: str) -> List[str]:
        """Fix the tokenization due to the `obj` and `subj` marker inserted
        in the middle of a word.
        Example:
            >>> text = "<obj>조지 해리슨</obj>이 쓰고 <subj>비틀즈</subj>가"
            >>> tokens = ['<obj>', '조지', '해리', '##슨', '</obj>', '이', '쓰', '##고', '<subj>', '비틀즈', '</subj>', '가']
            >>> fix_tokenization_error(text)
            ['<obj>', '조지', '해리', '##슨', '</obj>', '##이', '쓰', '##고', '<subj>', '비틀즈', '</subj>', '##가']

        Only support for BertTokenizerFast
        If you use bbpe, change code!
        """
        batch_encoding = tokenizer._tokenizer.encode(text)
        tokens = batch_encoding.tokens

        # subject
        if text[text.find(subject_end_marker) + len(subject_end_marker)] != " ":
            space_idx = tokens.index(subject_end_marker) + 1
            if tokenizer_type == "xlm-sp":
                if tokens[space_idx] == "▁":
                    tokens.pop(space_idx)
                elif tokens[space_idx].startswith("▁"):
                    tokens[space_idx] = tokens[space_idx][1:]
            elif tokenizer_type == "bert-wp":
                if not tokens[space_idx].startswith("##") and "가" <= tokens[space_idx][0] <= "힣":
                    tokens[space_idx] = "##" + tokens[space_idx]

        # object
        if text[text.find(object_end_marker) + len(object_end_marker)] != " ":
            space_idx = tokens.index(object_end_marker) + 1
            if tokenizer_type == "xlm-sp":
                if tokens[space_idx] == "▁":
                    tokens.pop(space_idx)
                elif tokens[space_idx].startswith("▁"):
                    tokens[space_idx] = tokens[space_idx][1:]
            elif tokenizer_type == "bert-wp":
                if not tokens[space_idx].startswith("##") and "가" <= tokens[space_idx][0] <= "힣":
                    tokens[space_idx] = "##" + tokens[space_idx]
        
        return tokens    

    tokens = fix_tokenization_error(examples["text"])

    return {
        "input_ids": tokenizer.convert_tokens_to_ids(tokens),
        "tokenized": tokens,
    }


def convert_type_example_to_features(
    examples,
    tokenizer,
) -> Dict[str, List[Any]]:

    subject_entity = examples["subject_entity"]
    object_entity = examples["object_entity"]

    subj_type = subject_entity["type"]
    obj_type = object_entity["type"]

    subject_end_marker   = f"</subj:{subj_type}>"
    object_end_marker    = f"</obj:{obj_type}>"

    tokenizer_type = check_tokenizer_type(tokenizer)

    def fix_tokenization_error(text: str) -> List[str]:
        """Fix the tokenization due to the `obj` and `subj` marker inserted
        in the middle of a word.
        Example:
            >>> text = "<obj>조지 해리슨</obj>이 쓰고 <subj>비틀즈</subj>가"
            >>> tokens = ['<obj>', '조지', '해리', '##슨', '</obj>', '이', '쓰', '##고', '<subj>', '비틀즈', '</subj>', '가']
            >>> fix_tokenization_error(text)
            ['<obj>', '조지', '해리', '##슨', '</obj>', '##이', '쓰', '##고', '<subj>', '비틀즈', '</subj>', '##가']

        Only support for BertTokenizerFast
        If you use bbpe, change code!
        """
        batch_encoding = tokenizer._tokenizer.encode(text)
        tokens = batch_encoding.tokens

        # subject
        if text[text.find(subject_end_marker) + len(subject_end_marker)] != " ":
            space_idx = tokens.index(subject_end_marker) + 1
            if tokenizer_type == "xlm-sp":
                if tokens[space_idx] == "▁":
                    tokens.pop(space_idx)
                elif tokens[space_idx].startswith("▁"):
                    tokens[space_idx] = tokens[space_idx][1:]
            elif tokenizer_type == "bert-wp":
                if not tokens[space_idx].startswith("##") and "가" <= tokens[space_idx][0] <= "힣":
                    tokens[space_idx] = "##" + tokens[space_idx]

        # object
        if text[text.find(object_end_marker) + len(object_end_marker)] != " ":
            space_idx = tokens.index(object_end_marker) + 1
            if tokenizer_type == "xlm-sp":
                if tokens[space_idx] == "▁":
                    tokens.pop(space_idx)
                elif tokens[space_idx].startswith("▁"):
                    tokens[space_idx] = tokens[space_idx][1:]
            elif tokenizer_type == "bert-wp":
                if not tokens[space_idx].startswith("##") and "가" <= tokens[space_idx][0] <= "힣":
                    tokens[space_idx] = "##" + tokens[space_idx]
        
        return tokens    

    tokens = fix_tokenization_error(examples["text"])

    return {
        "input_ids": tokenizer.convert_tokens_to_ids(tokens),
        "tokenized": tokens,
    }

  
# ref: https://github.com/KLUE-benchmark/KLUE-baseline/blob/8a03c9447e4c225e806877a84242aea11258c790/klue_baseline/data/utils.py#L92
def check_tokenizer_type(tokenizer: PreTrainedTokenizer) -> str:
    """Checks tokenizer type.
    In KLUE paper, we only support wordpiece (BERT, KLUE-RoBERTa, ELECTRA) & sentencepiece (XLM-R).
    Will give warning if you use other tokenization. (e.g. bbpe)
    # SentencePiece
      - XLMRobertaTokenizer : ['▁충', '무', '공', '▁이', '순', '신', '은', '▁조선', '▁중', '기의', '▁무', '신', '이다', '.']
      - kogpt2-base-v2      : ['▁충', '무공', '▁이순', '신은', '▁조선', '▁중기의', '▁무신', '이다.']
      - KoBART-base         : ['▁충', '무공', '▁이순', '신은', '▁조선', '▁중기의', '▁무신', '이다.']
    # BertTokenizer(Char-level, WordPiece)
      - KLUE(BERT, RoBERTa) : ['충무', '##공', '이순신', '##은', '조선', '중기', '##의', '무신', '##이다', '.']
      - KcELECTRA           : ['충무', '##공', '이순신', '##은', '조선', '중기', '##의', '무신', '##이다', '.']
      - KcBERT              : ['충', '##무', '##공', '이순신', '##은', '조선', '중', '##기의', '무신', '##이다', '.']
    """
    if isinstance(tokenizer, transformers.XLMRobertaTokenizer) or \
       isinstance(tokenizer, transformers.XLMRobertaTokenizerFast) or \
       tokenizer.name_or_path == "skt/kogpt2-base-v2" or \
       tokenizer.name_or_path == "KoBART-base":
        logger.info(f"Using {type(tokenizer).__name__} for fixing tokenization result")
        return "xlm-sp"  # Sentencepiece xlm-sp or gpt-sp or bart-sp
    elif isinstance(tokenizer, transformers.BertTokenizer) or \
         isinstance(tokenizer, transformers.BertTokenizerFast):
        logger.info(f"Using {type(tokenizer).__name__} for fixing tokenization result")
        return "bert-wp"  # Wordpiece (including BertTokenizer & ElectraTokenizer)
    else:
        logger.warn(
            "If you are using other tokenizer (e.g. bbpe), you should change code in `fix_tokenization_error()`"
        )
        return "other"
      
      
def get_entity_embedding(
    examples,
    tokenizer,
    subject_start_marker: str,
    subject_end_marker: str,
    object_start_marker: str,
    object_end_marker: str
) -> Dict[str, List[Any]]:

    subj_start_id = tokenizer.convert_tokens_to_ids(subject_start_marker)
    subj_end_id = tokenizer.convert_tokens_to_ids(subject_end_marker)
    obj_start_id = tokenizer.convert_tokens_to_ids(object_start_marker)
    obj_end_id = tokenizer.convert_tokens_to_ids(object_end_marker)

    entity_ids = []
    is_entity = False
    for input_id in examples["input_ids"]:
        if input_id in [subj_end_id, obj_end_id]:
            is_entity = False
        entity_id = 1 if is_entity else 0
        entity_ids.append(entity_id)
        if input_id in [subj_start_id, obj_start_id]:
            is_entity = True

    return {"entity_ids": entity_ids,}