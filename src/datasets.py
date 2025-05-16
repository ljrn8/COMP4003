import numpy as np
import pandas as pd 
import config


def load_preprocess_unsw(root=config.UNSWNB15_ROOT, shuffle_seed=None):
    
    testing_set = pd.read_csv(root / 'Training and Testing Sets' / 'UNSW_NB15_testing-set.csv')
    training_set = pd.read_csv(root / 'Training and Testing Sets' / 'UNSW_NB15_training-set.csv')
    LIST_EVENTS = pd.read_csv(root / 'UNSW-NB15_LIST_EVENTS.csv')
    
    NB15_1 = pd.read_csv(root / 'UNSW-NB15_1.csv')
    NB15_2 = pd.read_csv(root/ 'UNSW-NB15_2.csv')
    NB15_3 = pd.read_csv(root / 'UNSW-NB15_3.csv')
    NB15_4 = pd.read_csv(root / 'UNSW-NB15_4.csv')
    NB15_features = pd.read_csv(root / 'NUSW-NB15_features.csv', encoding='cp1252')
    
    NB15_1.columns = NB15_features['Name'] 
    NB15_2.columns = NB15_features['Name'] 
    NB15_3.columns = NB15_features['Name'] 
    NB15_4.columns = NB15_features['Name'] 
    
    train_df = pd.concat([NB15_1, NB15_2, NB15_3, NB15_4], ignore_index=True)
    if shuffle_seed:
        train_df = train_df.sample(
            frac=1, random_state=shuffle_seed).reset_index(drop=True) 

    train_df = train_df.drop_duplicates()
    
    # NOTE incomplete
    return train_df
    
    