import pandas as pd 
from icecream import ic 
import os 



RESULTS_DIR = '/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/datasets/results'

# Combine all 384
if __name__ == "__main__":


    MODELS = ['PIX'] # ,'GIT'] # ,'PIX']
    SIZE = '384'

    df = pd.DataFrame(columns=['qtype','precision','recall','fscore','average','size','model','model_category'])

    for model in MODELS:
        ic(model)
        # ROOT_DIR = '/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-' + model + '-hf/results-ddp/384a/'
        ROOT_DIR = '/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/models-' + model + '-hf/results-ddp/' + SIZE  + 'a'

        for f in os.listdir(ROOT_DIR):
            ic(f)
            model_df = pd.read_csv(os.path.join(ROOT_DIR,f,'summary_report.csv'))
            #ic(model_df.head(4))
            df = pd.concat([df,model_df])

        ic(df.shape)
        file_name = model + '-' + SIZE + '-final-summary-report.csv'

        df.to_csv(os.path.join(RESULTS_DIR,file_name),index=None)