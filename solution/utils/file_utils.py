# TODO
# task별로 나뉘어서 util 파일 관리

CONFIG_FILE_NAME = 'config.json'
PYTORCH_MODEL_NAME = 'pytorch_model.bin'

SUBJECT_ENTITIES = ['PER', 'ORG']
OBJECT_ENTITIES = ['PER', 'ORG', 'LOC', 'DAT', 'POH', 'NOH']

RELATION_CLASS = [
    'no_relation', 
    'org:top_members/employees',
    'org:members',
    'org:product',
    'per:title',
    'org:alternate_names',
    'per:employee_of',
    'org:place_of_headquarters',
    'per:product',
    'org:number_of_employees/members',
    'per:children',
    'per:place_of_residence', 
    'per:alternate_names',
    'per:other_family',
    'per:colleagues',
    'per:origin', 
    'per:siblings',
    'per:spouse',
    'org:founded',
    'org:political/religious_affiliation',
    'org:member_of',
    'per:parents',
    'org:dissolved',
    'per:schools_attended',
    'per:date_of_death', 
    'per:date_of_birth',
    'per:place_of_birth',
    'per:place_of_death',
    'org:founded_by',
    'per:religion'
]
       
NUM_CLASSES = len(RELATION_CLASS)
IDX2LABEL = {idx: label for idx, label in enumerate(RELATION_CLASS)}
LABEL2IDX = {label: idx for idx, label in enumerate(RELATION_CLASS)}

MARKERS = dict(
    subject_start_marker="<subj>",
    subject_end_marker="</subj>",
    object_start_marker="<obj>",
    object_end_marker="</obj>",
)

TYPE_MARKERS = dict(
    subject_start_per_marker="<subj:PER>",
    subject_start_org_marker="<subj:ORG>",
    subject_end_per_marker="</subj:PER>",
    subject_end_org_marker="</subj:ORG>",
    object_start_per_marker="<obj:PER>",
    object_start_org_marker="<obj:ORG>",
    object_start_loc_marker="<obj:LOC>",
    object_start_dat_marker="<obj:DAT>",
    object_start_poh_marker="<obj:POH>",
    object_start_noh_marker="<obj:NOH>",
    object_end_per_marker="</obj:PER>",
    object_end_org_marker="</obj:ORG>",
    object_end_loc_marker="</obj:LOC>",
    object_end_dat_marker="</obj:DAT>",
    object_end_poh_marker="</obj:POH>",
    object_end_noh_marker="</obj:NOH>",
)

# https://github.com/Saintfe/RECENT/blob/master/SpanBERT/code/run_tacred_multiple.py
TYPE_PAIR_TO_HEAD_ID = {
    "ORG_ORG": 0,
    "ORG_PER": 1,
    "ORG_DAT": 2,
    "ORG_LOC": 3,
    "ORG_POH": 4,
    "ORG_NOH": 5,
    "PER_ORG": 6,
    "PER_PER": 7,
    "PER_DAT": 8,
    "PER_LOC": 9,
    "PER_POH": 10, 
    "PER_NOH": 5,
    "LOC_DAT": 2,
}

HEAD_ID_TO_HEAD_LABELS = {
    0: # ORG_ORG
    {
        'no_relation': 0,
        'org:founded_by': 1,
        'org:members': 2,
        'org:political/religuous_affiliation': 3,
        'org:member_of': 4,     
    },
    1: # ORG_PER
    {
        'no_relation': 0,
        'org:founded_by': 1,
        'org:top_members/employees': 2,
        
    },
    2: # ORG_DAT
    {
        'no_relation': 0,
        'org:founded': 1,
        'org:dissolved': 2,
        
    },
    3: # ORG_LOC
    {
        'no_relation': 0,
        'org:members': 1,
        'org:member_of': 2,
        'org:place_of_headquarters': 3,
    },
    4: # ORG_POH
    {
        'no_relation': 0,
        'org:product': 1,
        'org:alternate_names': 2,
        'org:top_members/employees': 3,

    },
    5: # ORG_NOH
    {
        'no_relation': 0,
        'org:number_of_employees/members': 1,
    },
    6: # PER_ORG
    {
        'no_relation': 0,
        'per:employee_of': 1,
        'per:schools_attended': 2,
        'per:origin': 3,
        'per:religion': 4,
    },
    7: # PER_PER
    {
        'no_relation': 0,
        'per:colleagues': 1,
        'per:spouse': 2,
        'per:children': 3,
        'per:parents': 4,
        'per:other_family': 5,
        'per:siblings': 6,
    },
    8: # PER_DAT
    {
        'no_relation': 0,
        'per:date_of_birth': 1,
        'per:date_of_death': 2,
    },
    9: # PER_LOC
    {
        'no_relation': 0,
        'per:place_of_residence': 1,
        'per:place_of_birth': 2,
        'per:place_of_death': 3,
        'per:origin': 4,
    },
    10: # PER_POH
    {
        'no_relation': 0,
        'per:title': 1,
        'per:product': 2,
        'per:alternate_names': 3,
        
    },
}


HEAD_ID_TO_LABEL_ID = [
    # ORG_ORG
    [0, 28, 2, 19, 20],
    # ORG_PER
    [0, 28, 1],
    # ORG_DAT
    [0, 18, 22],
    # ORG_LOC
    [0, 2, 20, 7],
    # ORG_POH
    [0, 3, 5, 1],
    # ORG_NOH
    [0, 9],
    # PER_ORG
    [0, 6, 23, 15, 29],
    # PER_PER
    [0, 14, 17, 10, 21, 13, 16],
    # PER_DAT
    [0, 25, 24],
    # PER_LOC
    [0, 11, 26, 27, 15],
    # PER_POH
    [0, 4, 8, 12],
]


class TAPT:
    label = None
    id2label = None
    label2id = None
    num_labels = 2 # set default
    markers = MARKERS

    
class KLUE_RE:
    label = RELATION_CLASS
    subject_entity = SUBJECT_ENTITIES
    object_entity = OBJECT_ENTITIES
    id2label = {idx: label for idx, label in enumerate(RELATION_CLASS)}
    label2id = {label: idx for idx, label in enumerate(RELATION_CLASS)}
    num_labels = NUM_CLASSES
    markers = MARKERS
    
    
class KLUE_RE_TYPE(KLUE_RE):
    markers = TYPE_MARKERS
    
    
class RECENT:
    label = RELATION_CLASS
    subject_entity = SUBJECT_ENTITIES
    object_entity = OBJECT_ENTITIES
    id2label = {idx: label for idx, label in enumerate(RELATION_CLASS)}
    label2id = {label: idx for idx, label in enumerate(RELATION_CLASS)}
    num_labels = NUM_CLASSES
    markers = MARKERS
    type_pair_to_head_id = TYPE_PAIR_TO_HEAD_ID
    head_id_to_head_labels = HEAD_ID_TO_HEAD_LABELS
    head_id_to_label_id = HEAD_ID_TO_LABEL_ID


TASK_INFOS_MAP = {
    "klue_re": KLUE_RE,
    "tapt": TAPT,
    "klue_re_type": KLUE_RE_TYPE,
    "recent": RECENT,
}