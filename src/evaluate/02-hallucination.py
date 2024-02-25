import pandas as pd 
import sys 
import os 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from icecream import ic 
import re 
import os 
import numpy as np 
from tqdm import tqdm 
import ast

if __name__ == "__main__":
    MODEL="LLaVA"
    PREDICTION_FILE="predictions-final.csv"
    #exp_list = ['ocr-post','base','desc','bbox','ocr-pre']
    exp_list = ['bbox-yolo']
    
    # MODEL='BLIP'  # LLaVA
    # PREDICTION_FILE="predictions.csv"
    # exp_list = ['base','base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all'] # BLIP


    # MODEL='PIX'  # LLaVA
    # PREDICTION_FILE="predictions.csv"
    # exp_list = ['base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all']

    # MODEL='GIT'  # LLaVA
    # PREDICTION_FILE="predictions.csv"
    # exp_list = ['base','base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all'] # GIT

    # MODEL='PIX' # BLIP
    # PREDICTION_FILE="predictions.csv"
    # exp_list = ['base-lr','desc','ocr-pre','ocr-post','wce','bbox','bbox-segment','all','bbox-yolo'] # BLIP


    ROOT_DIR = '/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-' + MODEL  + '-hf'
    SIZE = '384' # 576 
    
    
    
    hs_vqa_list,hs_cnt_list,hs_cntcomplex_list,hs_val_list,hs_obj_list = [],[],[],[],[]
    op_dir = 'results-ddp/'  + SIZE + 'a'

    for EXP in exp_list:
        ic(EXP)
        
        results_dir = 'results-ddp/'  + SIZE + 'a/'+ EXP
        df = pd.read_csv(os.path.join((ROOT_DIR),results_dir,PREDICTION_FILE))
        #ic(df.head(2))
        
        # #if MODEL in ['BLIP','GIT','PIX']:
        df['prediction'] = df.apply(lambda row : str(row.prediction).split(".")[0] if row.qtype in ['count-complex','count']  else str(row.prediction),axis=1)
        df['answer'] = df.apply(lambda row : row.answer.split(".")[0] if row.qtype in ['count-complex','count']  else row.answer,axis=1)
        #ic(df.info())

        # df['cnt_over']  = np.where(df['prediction'] > df['answer'],df['prediction']-df['answer'],0)
        
        df = df[df['qtype'].isin(['count','count-complex','value','junction'])] # 'count','count-complex'

        #total_cnt = 0 
        df['cnt_over'] = 0
        for j,row in tqdm(df.iterrows(),total=df.shape[0]):
            #ic(row)

            if row['qtype'] in ['junction']:
                #total+=1
                cnt_over = 0 
                if MODEL == 'LLaVA':
                    l = row['raw_prediction'].split()
                else:
                    l = str(row['prediction']).split()

                oovocab = ['circle','square','A','B','D','F','triangle','carlin','nano','peizo-keeper','trigger','Snake Snake Detector','1','2','3','4','5','6','7','8']
                for item in oovocab:
                    if item in l:
                        cnt_over +=1 

            elif row['qtype'] in ['count','count-complex']:
                #total_cnt+=1
                cnt_over  = 0 
                #ic(row['prediction'],row['answer'])
                #if row['prediction'] in ['yes','no','xor','or','resistor','not','gnd','nor',"['1","['1ωa']","['1η']","['2v']","['100']","['1ohm', '1ohm']", "['1ohm', '3ohm']","['25v']","['1ohm', '2ohm', '3ohm']"]:

                l = [str(i) for i in range(0,100)]
                if row['prediction'] not in l:  
                    p = 0
                else:
                    p = int(row['prediction'])
                a = int(row['answer'])
                if p > a:
                    cnt_over = p - a

            elif row['qtype'] == 'value': # Ocr
               #total+=1 
                cnt_over = 0 

                if MODEL == 'LLaVA':
                    lp = len(list(ast.literal_eval(row['prediction'])))
                    la = len(list(ast.literal_eval(row['answer'])))
                else:
                    lp = len(row['prediction'][1:-1].split(','))
                    la = len(row['answer'][1:-1].split(","))
                #     lp = len(list(row['prediction']))
                #     la = len(list(row['answer']))
                #ic(lp,la)
                if lp > la:
                    cnt_over = lp - la
                

            df.at[j,'cnt_over'] = cnt_over 

        # HS score overall
        hs_cnt = df['cnt_over'][df['qtype']=='count'].sum()/df[df['qtype']=='count'].shape[0]
        ic(hs_cnt,df['cnt_over'][df['qtype']=='count'].sum(),df[df['qtype']=='count'].shape[0])

        # HS count complex 
        hs_cntcomplex = df['cnt_over'][df['qtype']=='count-complex'].sum()/df[df['qtype']=='count-complex'].shape[0]
        ic(hs_cntcomplex,df['cnt_over'][df['qtype']=='count-complex'].sum(),df[df['qtype']=='count-complex'].shape[0])

        # HS value
        hs_value = df['cnt_over'][df['qtype']=='value'].sum()/df[df['qtype']=='value'].shape[0]
        ic(hs_value,df['cnt_over'][df['qtype']=='value'].sum(),df[df['qtype']=='value'].shape[0])

        hs_obj = df['cnt_over'][df['qtype']=='junction'].sum()/df[df['qtype']=='junction'].shape[0]
        ic(hs_obj,df['cnt_over'][df['qtype']=='junction'].sum(),df[df['qtype']=='junction'].shape[0])

        ## hs-overall
        hs_vqa = (hs_cnt + hs_cntcomplex + hs_value + hs_obj)/4 
        hs_vqa_list.append(hs_vqa)
        hs_cnt_list.append(hs_cnt)
        hs_cntcomplex_list.append(hs_cntcomplex) 
        hs_val_list.append(hs_value)
        hs_obj_list.append(hs_obj)

    
    res_df = pd.DataFrame(zip(exp_list,hs_vqa_list,hs_cnt_list,hs_cntcomplex_list,hs_val_list,hs_obj_list),columns=['exp-name','hs-vqa','hs-count','hs-countcomplex','hs-val','hs-obj'])
    res_df['model'] = MODEL
    ic(res_df)
    res_df.to_csv(os.path.join(ROOT_DIR,op_dir,"hallucination-scores.csv"),index=None)



        
