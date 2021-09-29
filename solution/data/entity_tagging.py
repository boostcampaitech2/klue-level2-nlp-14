import torch
from functools import partial
from typing import Tuple, List, Any, Dict


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


def convert_example_to_features(
    examples, 
    tokenizer,
    subject_start_marker: str,
    subject_end_marker: str,
    object_start_marker: str,
    object_end_marker: str
) -> Dict[str, List[Any]]:
    
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
            # tokenizer_type == "bert-wp"
            if not tokens[space_idx].startswith("##") and "가" <= tokens[space_idx][0] <= "힣":
                tokens[space_idx] = "##" + tokens[space_idx]

        # object
        if text[text.find(object_end_marker) + len(object_end_marker)] != " ":
            space_idx = tokens.index(object_end_marker) + 1
            # tokenizer_type == "bert-wp"
            if not tokens[space_idx].startswith("##") and "가" <= tokens[space_idx][0] <= "힣":
                tokens[space_idx] = "##" + tokens[space_idx]
        
        return tokens    
    
    tokens = fix_tokenization_error(examples["text"])
    
    return {
        "input_ids": tokenizer.convert_tokens_to_ids(tokens),
        "tokenized": tokens,
    }