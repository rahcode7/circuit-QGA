import argparse
import os
import re
import string
import time
import json
import inflect 
import pandas as pd 
import numpy
from word2number import w2n
from icecream import ic 
from tqdm import tqdm 
# commands 
#python post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/results/llava/384a/base
#python models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-InstructBLIP-hf/results-ddp/384a --model InstructBLIP --exp_name base
#python models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-InstructBLIP-hf/results-ddp/384a --model InstructBLIP --exp_name bbox
# python models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-InstructBLIP-hf/results-ddp/384a --model InstructBLIP --exp_name ocr-post
# python models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-InstructBLIP-hf/results-ddp/384a --model InstructBLIP --exp_name ocr-pre
# python models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-InstructBLIP-hf/results-ddp/384a --model InstructBLIP --exp_name desc

# python models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-InstructBLIP-hf/results-ddp/384a --model InstructBLIP --exp_name bbox-segment
# python models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-InstructBLIP-hf/results-ddp/384a --model InstructBLIP --exp_name bbox-yolo
# python models-hf/models-InstructBLIP-hf/post-process.py --prediction_dir /Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-InstructBLIP-hf/results-ddp/384a --model InstructBLIP --exp_name bbox-segment-yolo

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--prediction_dir', help='directory')
    parser.add_argument('--model', help='model')
    parser.add_argument('--exp_name', help='exp')
   
    args = parser.parse_args()

    RESULTS_DIR = args.prediction_dir
    EXP_NAME = args.exp_name 
    MODEL = args.model 


    input_file = os.path.join(RESULTS_DIR,EXP_NAME,'predictions.json')
    df = pd.read_json(input_file, lines=True)
    print(df.head(4))

    # If desc, merge with base predictions 
    # if EXP_NAME == 'desc':
    #     df2 = pd.read_json('/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-LLaVa-hf/results-ddp/384a/base/predictions.json', lines=True)
    #     df2 = df2[df2.qtype.isin(['position','value','junction'])]
    #     ic(df2['qtype'].value_counts())
    #     df = pd.concat([df,df2],ignore_index=True)
    # ic(df.shape)

    p = inflect.engine()

    nums = [i for i in range(0,51)]
    nums_words = [p.number_to_words(i).replace("-"," ") for i in nums]
    nums_words = nums_words[::-1]
    
    # For each row check question type 
    # extract any number or the answer 

    for j,row in tqdm(df.iterrows(),total=df.shape[0]):

        answer,pred,qtype = row['answer'],row['prediction'],row['qtype']
        pred = re.sub(r'\W+', ' ',pred)
        #ic(pred)

        # if strin is empty
        if not pred:
            pred_final,pred_new = "",""
            df.at[j,'pred_final'] = pred_final
            df.at[j,'pred_new'] = pred_new
            continue

        if pred[-1] == ".":
            pred = pred[:-1]
        #ic(pred)
        sl = pred.lower().split(" ")
        #ic(sl)
        
        if qtype == 'count' or qtype == 'count-complex':
            pred_final,pred_new = -1,-1

            # FOR INSTRUCT-BLIP  
            if MODEL == 'InstructBLIP':   
                # pred_final = pred
                # pred_new = pred 
                for num in nums:
                    if str(num) in pred:
                        pred_final = num  
                        pred_new = num      
            elif MODEL == 'LLaVA':            
                # There are two multi-cell batteries connected directly to the left of C1.
                # There are two capacitor-unpolarized connected directly to the left of C19.
                # Yes, the image shows a circuit diagram with a total of 14 diode light emittings.
                # The image shows a circuit diagram with a number of resistors, but without knowing the specific details of the diagram, it is not possible to provide an accurate count of the resistors.
                # There are two components in the circuit that function as current-sources.
                for num in nums:
                    if " " + str(num) + " " in pred:
                        pred_final = num 

                # Check "forty five" in the string
                for i ,word in enumerate(nums_words):
                    if word in pred:
                        pred_final = w2n.word_to_num(word)
                
                if "single" in pred:
                    pred_final = w2n.word_to_num("one") 

            #ic(pred,pred_final)
            df.at[j,'pred_final'] = int(pred_final)
            df.at[j,'pred_new'] = int(pred_new)
            # check for numbers 
        elif qtype == 'junction':
            pred_final,pred_new = "",""
            pass 
            # answer yes or no - first word 
            pred_final = sl[0]

            df.at[j,'pred_final'] = pred_final
            df.at[j,'pred_new'] = pred_new

        elif qtype == 'position':
            pred_final,pred_new,pred_new_list = "","",[]
            # search answer keyword in "is a {answer}"
            "The circuit symbol at the extreme top is a lightning bolt."
            "The circuit symbol on the leftmost side is a circle with a line through it."
           
            # check next 1 or words of answer 
            if "is" in sl:
                    if "a" in sl:
                        i = sl.index("a")
                        if sl.index("is") + 1 == sl.index("a"):
                            pred_new  = " ".join(sl[i+1:])
                            pred_new_list = sl[i+1:]
                    else:
                        i = sl.index("is")
                        pred_new  = " ".join(sl[i+1:])
                        pred_new_list = sl[i+1:]
            elif "represents" in sl:
                i = sl.index("a")
                if sl.index("represents") + 1 == sl.index("a"):
                    pred_new  = " ".join(sl[i+1:])
                    pred_new_list = sl[i+1:]
            else:
                pass 
                
            df.at[j,'pred_final'] = pred_final
            df.at[j,'pred_new'] = pred_new

            # Now Check if answer exist in this segment, then extract answer and write as answer else write the segment
            # answers - integrated_circuit.voltage_regulator,resistor.adjustable, resistor 

            flag=False
            if len(pred_new_list) !=0:
                for item in answer.split("."):
                    if item in pred_new_list:
                        flag=True
                    else:
                        flag=False
                if flag:
                    pred_final = answer.strip()
                else:
                    pred_final = pred_new.strip()
            else:
                pass

            df.at[j,'pred_final'] = pred_final
            df.at[j,'pred_new'] = pred_new

        elif qtype == 'value':
            pred_final,pred_new = [],[]
            for s in sl:
                if any(letter.isdigit() for letter in str(s)):
                    pred_final.append(s)   
            # "The resistors in the image are showing a reading of 20 and 30."
            # "The voltmeter displays a reading of 15 volts."
            # The values displayed on the voltage-dc_ac are 220V and 30V.
            # The current value indicated on the resistors is 100mA.
            df.at[j,'pred_final'] = pred_final
            df.at[j,'pred_new'] = pred_new
            # "Unfortunately, I cannot provide the current reading of the resistor.adjustable as it is not visible in the image. The image only shows a diagram of a circuit board with various components, but the actual values of the components are not provided."
            # # collect number in a list      
        else:
            pass 
            
        #ic(pred_final,pred_new,qtype,answer)

    
    df.rename(columns={'prediction':'raw_prediction','pred_final':'prediction','pred_new':'pred_segment'},inplace=True)    
    df.to_csv(os.path.join(RESULTS_DIR,EXP_NAME,'predictions-final.csv'), index=None) # , lines=True)
    df.to_json(os.path.join(RESULTS_DIR,EXP_NAME,'predictions-final.json'), orient='records', lines=True)
