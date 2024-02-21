from torch.utils.data import Dataset,DataLoader
import os 
from PIL import Image
import json 
from icecream import ic 

class VQACircuitDataset(Dataset):
    def __init__(self,root_dir,Q_PATH,split) -> None:
        with open(Q_PATH, 'r') as f:
            self.qa = json.load(f)
        #self.image_dir = image_dir
        self.split = split  
        self.qa = [i for i in self.qa if i['splittype']==split]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.qa)

    def __getitem__(self,idx):
        file_name = self.qa[idx]["file"] + '.jpg'
        question = self.qa[idx]["question"]
        answer = self.qa[idx]["answer"]
        qtype = self.qa[idx]["qtype"]

        #ic(qtype)
        if qtype == "junction":
            # try:    
            image_path = os.path.join(self.root_dir,'model-inputs-jn',self.split,file_name)
            image = Image.open(image_path).convert('RGB')   
            # except:
            #     return None
        else:
            image_path = os.path.join(self.root_dir,'model-inputs',self.split,file_name)
            image = Image.open(image_path).convert('RGB')   

        return image,question,answer,qtype

def vqa_collate_fn(batch):
    len_batch = len(batch)
    image_list, question_list, answer_list,qtype_list = [], [], [],[]
    #ic(batch)

    batch = list(filter (lambda x:x is not None, batch))
    #ic(batch)
    for image, question, answer,qtype  in batch:
        image_list.append(image)
        question_list.append(question)    
        answer_list.append(str(answer))
        qtype_list.append(qtype)

    return image_list,question_list,answer_list,qtype_list
