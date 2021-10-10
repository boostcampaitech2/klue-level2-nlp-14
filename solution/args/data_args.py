from typing import List
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    name: str = field(
        default="jinmang2/load_klue_re", metadata={"help": "Dataset name or path"},
    )
    revision: str = field(
        default="v1.0.0", metadata={"help": "Dataset version"},
    )
    data_cache_dir: str = field(
        default=None, metadata={"help": "Dataset cache directory path"},
    )
    collator_name: str = field(
        default="default", metadata={"help": "Data collator name"},
    )
    prep_pipeline_name: str = field(
        default="entity_tagging",
        metadata={"help": "Preprocessing pipeline name. "
                          "You can find this object on `solution/data/__init__.py`"},
    )
    max_length: int = field(
        default=256, metadata={"help": "Max token length"},
    )
    augment: str = field(
        default="original", metadata={"help": "Augmented dataset name"},
    )
