from typing import List
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    name: str = field(
        default="jinmang2/load_klue_re", metadata={"help": ""},
    )
    revision: str = field(
        default="v1.0.0", metadata={"help": ""},
    )
    data_cache_dir: str = field(
        default=None, metadata={"help": ""},
    )
    collator_name: str = field(
        default="default", metadata={"help": ""},
    )
    prep_pipeline_name: str = field(
        default="entity_tagging", metadata={"help": ""},
    )
    max_length: int = field(
        default=256, metadata={"help": "max token length"},
    )
    augment: str = field(
        default="original", metadata={"help": ""},
    )
    