# INFERENCE
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering
from datasets import VQACircuitDataset, vqa_collate_fn
from torch.utils.data import DataLoader
import torch
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from tqdm import tqdm
import os
from icecream import ic
import pandas as pd
import os 
import argparse

ROOT_DIR = os.getcwd()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_dir', help='Test  directory')
    parser.add_argument('--test_batch_size',type=int, help='Check directory')
    parser.add_argument('--checkpoint_dir', help='Check directory')
    parser.add_argument('--results_dir', help='Results directory')
    parser.add_argument('--question_dir', help='Q directory')
    parser.add_argument('--machine_type', help='dp or ddp or single or cpu')

    args = parser.parse_args()

    CHECKPOINTS_DIR = args.checkpoint_dir
    TEST_IMAGE_DIR =args.test_dir
    OUTPUT_PATH = args.results_dir 
    Q_PATH = args.question_dir
    TEST_BATCH_SIZE=args.test_batch_size
    MACHINE_TYPE=args.machine_type

    # Create results folder
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    test_dataset = VQACircuitDataset(TEST_IMAGE_DIR, Q_PATH, split='test')
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=TEST_BATCH_SIZE, collate_fn=vqa_collate_fn)

    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base",ignore_mismatched_sizes=True)
    ic("model loaded")

    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ic(device)
    if MACHINE_TYPE=='dp':
        ic("model dp")
        model = DataParallel(model)
    elif MACHINE_TYPE == 'ddp':
        model = DDP(model)
    else:
        pass
        
    model.to(device)


    starting_name = 'checkpoint'
    file_list = os.listdir(CHECKPOINTS_DIR)
    ic(file_list)

    checkpoint_files = [filename for filename in file_list if filename.startswith(starting_name)]
    checkpoint_files.sort()

    checkpoint_file = checkpoint_files[-1]
    ic(checkpoint_file)

    checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,checkpoint_file))
    ic("checkpoint file loaded",checkpoint_file)

    # ic(checkpoint)
    model.load_state_dict(checkpoint['model'],strict=False)
    model.eval()

    # ic(model)
    total = 0
    accuracy_batch = 0.0

    col_list = ['splittype', 'question', 'answer', 'prediction','qtype']
    pred_df = pd.DataFrame(columns=col_list)

    for idx, (images, questions, answers,qtype) in enumerate(tqdm(test_dataloader)):
        total += len(answers)
        #ic(idx)
        inputs = processor(images=images, text=questions,
                           return_tensors="pt", padding=True)
        labels = processor(text=answers, return_tensors="pt",
                           padding=True).input_ids
        inputs["labels"] = labels

        inputs = inputs.to(device)
        labels = labels.to(device)

        if MACHINE_TYPE in  ['dp','ddp']:
            outputs = model.module.generate(**inputs)
        else:
            outputs = model.generate(**inputs)
    
        predicted = processor.batch_decode(outputs, skip_special_tokens=True)
        accuracy_batch += len([i for i,j in zip(predicted, answers) if i == j])

        # store answers in json or csv
        data = [questions, answers, predicted,qtype]
        #ic(data)
        df = pd.DataFrame(data).transpose()
        df.columns = ['question', 'answer', 'prediction','qtype']

        df['splittype'] = 'test'
        pred_df = pd.concat([pred_df, df])

    # #  Compute Accuracy
    test_accuracy = (100 * accuracy_batch / total)
    ic(test_accuracy, total)

    pred_df.to_csv(OUTPUT_PATH + '/predictions.csv', index=None)
