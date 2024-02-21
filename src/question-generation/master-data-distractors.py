import pandas as pd
import os 
from pathlib import Path
from icecream import ic 

DATA_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/questions/"



if __name__ == "__main__":

    #qtypes = ['count','count-complex','position','value']
    qtypes = ['count','position']
    dist_df_all = pd.DataFrame(columns=['splittype', 'file', 'question', 'answer', 'qtype', 'distractor1',
       'distractor2', 'distractor3', 'distractor4'])

    for qtype in qtypes:
        # q_path = os.path.join(DATA_PATH,qtype)
        # file_str = "Q-"+ qtype + ".csv"
        # ques_df = pd.read_csv(q_path+'/'+file_str)
        # print(ques_df.columns)

        # if 'symbol' not in ques_df.columns:
        #     ques_df['symbol'] = ""

        # postprocessing
        #if qtype == ""
        #if qtype == "value":
            # For resistors all ohm values or all values only subset Qs

            
        # Concat all Distractor files
        file_path = os.path.join(DATA_PATH,qtype,"Q-" + qtype + "-distractor" + ".csv")
        #print(file_path)
        file_path = Path(file_path)
        
        if file_path.is_file():
            dist_df = pd.read_csv(file_path)
            print(dist_df.shape)
            dist_df_all = pd.concat([dist_df_all,dist_df])

        ic(qtype,dist_df_all.shape)
        dist_df_all.to_csv(DATA_PATH+"all/"+"QA-distractors.csv",index=None)
        dist_df_all[['question','answer','file','splittype']].to_json(DATA_PATH+"/all/"+"master.json",orient="records")

