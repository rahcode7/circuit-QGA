
import pandas as pd
import os 
from pathlib import Path
from icecream import ic 
import numpy as np 
import json
import random 


DATA_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/questions/"
SAMPLES = 400


if __name__ == "__main__":
    # Qs
    df = pd.read_csv(DATA_PATH + "/all/"+"master_adv_ocr.csv")
    ic(df.head(2))

    #set.seed(42)
    #random.seed(42)


    ic(df.groupby(['splittype'])['file'].nunique())


    df_tr = df[df['splittype']=='train'].sample(int(SAMPLES*0.6),random_state=42)
    df_te = df[df['splittype']=='test'].sample(int(SAMPLES*0.2),random_state=42)
    df_val = df[df['splittype']=='val'].sample(int(SAMPLES*0.2),random_state=42)


    master_sample = pd.concat([df_tr,df_te,df_val])
    
    master_sample.to_csv(DATA_PATH+"/all/"+"master_adv_ocr_sample_100.csv",index=None)
    master_sample.to_json(DATA_PATH+"/all/"+"master_adv_ocr_sample_100.json",orient="records")