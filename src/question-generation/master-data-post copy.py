"""
Removing near duplicates
"""

import pandas as pd
import os 
from pathlib import Path
from icecream import ic 
import numpy as np 
import json


DATA_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/questions/"

if __name__ == "__main__":
    # Qs
    df = pd.read_csv(DATA_PATH + "/all/"+"master.csv")

    # Delete jpg if it exists 
    df['file'] = df['file'].apply(lambda x: x.split(".jpg")[0] if x.count(".jpg") >1 else x)

    ic(df.head(2))

    ic(df.groupby(['splittype'])['file'].nunique())

    # Check near dups in master Qs and count splits 
    with open("/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/processed/duplicate_unique_1.json") as f:
        dups = json.load(f)


    ic(df.info())
    df['file_hasdup'] = df['file'].apply(lambda x : 1 if x + ".jpg" in dups.keys() else 0)
    ic(df.groupby(['splittype','file_hasdup'])['file'].nunique())

    # identify dup files
    #ic(list(dups.values()))
    ic(df['file'].head(10))
    df['file'] = df['file'].astype("string")
    df['file_isdup'] = df['file'].apply(lambda x : 1 if x + ".jpg" in list(dups.values()) else 0)
    ic(df.groupby(['splittype','file_isdup'])['file'].nunique())

    ic(df['file_isdup'].value_counts())
    # Remove these duplicate files from Question bank 
    ic(df[df['file_isdup']==1])
    ic(df.shape)
    master_df = df[df['file_isdup']==0]
    ic(master_df.shape)
    ic(master_df.groupby(['splittype','file_isdup'])['file'].nunique())

    master_df.drop(columns=['file_isdup','file_hasdup'],inplace=True)

    ic(master_df.info())
    # master_df.to_csv(DATA_PATH+"/all/"+"master.csv",index=None)
    # master_df.to_json(DATA_PATH+"/all/"+"master.json",orient="records")
    # ic("Qs written")

    






