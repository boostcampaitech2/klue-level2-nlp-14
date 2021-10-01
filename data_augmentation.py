import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple, Union, Dict, Text

def find_nth(string, substring, n):
    if (n == 1):
        return string.find(substring)
    else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)

def entity_prepro(sentence, entity):
    start_idx = find_nth(sentence, entity[0], entity[2])
    end_idx = start_idx + len(entity[0])
    
    p_entity = {
          "word" : entity[0],
          "start_idx" : start_idx,
          "end_idx" : end_idx,
          "type" : entity[1]
        }
    
    return p_entity


def data_organizing(
    sentence : Text,
    subjects : Tuple[str, str, int],
    objects : Tuple[str, str, int]
) -> Union[Text, Dict[str, str], Dict[str, str]]:
    
    p_subjects = entity_prepro(sentence, subjects)
    p_objects = entity_prepro(sentence, objects) 
    
    return [sentence, p_subjects, p_objects]

def augmentation(
    tagged_sentences : Union[List[Tuple[str, str]]]
) -> List[Union[str, Dict, Dict]]:
    
    tagged_sentence_word_cnt = []
    
    for sent in tagged_sentences: # 토큰별로 몇번째로 등장했는지 추가
        tmp = ''
        count_tagged = []
        for tok, tag in sent:
            count_tagged.append((tok, tag, tmp.count(tok)+1))
            tmp += tok
        tagged_sentence_word_cnt.append(count_tagged)
    
    print("Number of Data to aumgented :", len(tagged_sentence_word_cnt))
    
    augmented_data = []
    for tag_sent in tagged_sentence_word_cnt:
        org_sent = "".join([tok for tok, tag, _ in tag_sent])
        obj_list = [(tok, tag, cnt) for tok, tag, cnt in tag_sent if tag_map[tag]!='O']
        sbj_list = [(tok, tag, cnt) for tok, tag, cnt in obj_list if tag in ['PERSON', 'ORGANIZATION']]
        cand_list = [[org_sent, sbj, obj] for sbj in sbj_list for obj in obj_list if sbj!=obj]
        augmented_data.extend([data_organizing(sent, sbj, obj) for sent, sbj, obj in cand_list])
        
    print("Number of Augmented data :", len(augmented_data))
    
    return augmented_data

def main():
    using_tag = ['PERSON', 'LOCATION', 'ORGANIZATION', 'DATE', 'TIME', 'CITY']

    tag_map = {
        'PERSON' : 'PER',
        'LOCATION' : 'LOC',
        'ORGANIZATION' : 'ORG',
        'CITY' : 'LOC',
        'COUNTRY' : 'ORG', #ORG
        'ARTIFACT' : 'O',
        'DATE' : 'DAT',
        'TIME' : 'DAT',
        'CIVILIZATION' : 'O',
        'ANIMAL' : 'O',
        'PLANT' : 'O',
        'QUANTITY' : 'NOH',
        'STUDY_FIELD' : 'O',
        'THEORY' : 'O',
        'EVENT' : 'O', #ORG
        'MATERIAL' : 'O',
        'TERM' : 'O',
        'OCCUPATION' : 'O', #직업
        'DISEASE' : 'O',
        'O' : 'O',
    }
    with open('tagged_sentence.pickle', 'rb') as f:
        tagged_sentence = pickle.load(f)

    aug_data = augmentation(tagged_sentence)
    
    augmented_data = pd.DataFrame(aug_data)
    augmented_data.columns = ['sentence', 'subject_entity', 'object_entity']
    augmented_data['label'] = None
    augmented_data['source'] = 'augmented'
    
    augmented_data.to_csv("augmented_data.csv", index=False)

    with open('augmented_data.pickle', 'wb') as f:
        pickle.dump(augmented_data, f, pickle.HIGHEST_PROTOCOL)
        