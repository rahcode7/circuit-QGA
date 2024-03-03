# title	annotations	yt_url
# Balancing Business & Lifestyle "chapter annotations\nchapter name: Create a list\nvisual captions: In the video, we'll see then [Music] \nIMAGE:https://raw.githubusercontent.com/ceyxasm/vid-frames-dv1-p7/master/RzcHE7KJmoc/RzcHE7KJmoc/1.png\n

# tab separated
# How to measure for a new washing machine | Currys PC World  "chapter annotations\nchapter name: Intro\nvisual captions"  https://www.youtube.com/watch?v=dSUhX8JTC3I

import json 
import pandas as pd 
import os 
from icecream import ic 
import numpy as np

if __name__ == "__main__":

    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000
    pd.set_option('display.max_colwidth', None)  # or 199


    

    OP_PATH = "models-hf/gpt4v/datasets/desc"
    # Read master.json to get question,file for generation

    Q_PATH = "datasets/questions/all"
    FILE_NAME = "master_adv.json"
    df  =pd.read_json(os.path.join(Q_PATH,FILE_NAME))
    df = df[df['splittype']=='test']


    ic(df.head(2))
    ic(df.desc.isna().sum())
    
    # create 23 datasets in format question image_url 

    # https://rahcode7.github.io/d1_258_png.rf.150231c3566da1bc9946f23678909d2a.jpg
    #https://rahcode7.github.io/d4_part2_-8_png.rf.3dee55a06343bf2380802388bfda0abe.jpg
    
    
    # Update description, extract symbol 
    # def get_symbol(q,s):

    #     # get dict
        
    #     # with open("/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/src/utils/class_dict.json") as c:
    #     #     d = json.load(c)    
    #     return s 
    
    
    # df['symbol'] = df.apply(lambda x : get_symbol(x.question) if x.symbol is None else x.symbol)



    # Add desc to question 
    def desc(q,s,d):
        #ic(q,s,d)
        if d[-1]==".":
            #return("The following question is about an electrical component : " + s + "  .It cab be described as follows: " +  d + " Now, please provide an answer to the given question. " + q)
            return("Use the following description of the electrical component to answer the question: " +  d + " Now, respond to this question: " + q)
        else:
            return("Use the following description of the electrical component to answer the question: " +  d + ". Now, respond to this question: " + q)
        
    df['question'] = df.apply(lambda x : desc(x.question,x.symbol,x.desc) if x.desc is not None else x.question,axis=1)

    df = df.reset_index()
    #ic(df.head(4))
    df['id'] = df.index + 1
    df['image_url']  = df['file'].apply(lambda x: "https://rahcode7.github.io/"  + x + ".jpg")
    #print(df['image_url'].head(2))
    #ic(df.head(4))

    df = df[['id','file','question','image_url']]

    #ic(df['file'].nunique())

    NUMBER_OF_SPLITS = 24

    for i, new_df in enumerate(np.array_split(df,NUMBER_OF_SPLITS)):
        file_name = f"master{i}.txt"
        ic(file_name)
        with open(os.path.join(OP_PATH,file_name),"w") as fo:
            fo.write(new_df.to_csv(sep="\t",index=None))