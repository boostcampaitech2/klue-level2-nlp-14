import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold


def kfold_split(dataset, n_splits=5, fold=1, random_state=42):
    full_df = pd.DataFrame(dataset)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_indices, valid_indices) in enumerate(kfold.split(full_df, full_df.label), 1): # fold: [1, n_splits]
        if fold_idx == fold:
            print("="*20 + f" fold: {fold_idx} "+ "="*20)
            train_dataset = Dataset.from_pandas(full_df.iloc[train_indices])
            eval_dataset = Dataset.from_pandas(full_df.iloc[valid_indices])
            break
    return train_dataset, eval_dataset