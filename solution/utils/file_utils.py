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


TASK_INFOS_MAP = {
    "klue_re": KLUE_RE,
    "tapt": TAPT,
    "klue_re_type": KLUE_RE_TYPE,
}