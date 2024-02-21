
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering,BlipImageProcessor
from datasets import VQACircuitDataset, vqa_collate_fn
from torch.utils.data import DataLoader
import torch
import wandb
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import argparse
from utils.helpers import set_seed
from icecream import ic
import torch.nn as nn 
import logging
import time 
import random 
import gc


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=1,
                        type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=2e-5,
                        type=int, help='Learning rate')
    parser.add_argument('--train_batch_size', default=2,
                        type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=2,
                        type=int, help='val batch size')
    parser.add_argument('--test_batch_size', default=2,
                        type=int, help='test batch size')
    parser.add_argument('--train_dir', help='train directory')
    parser.add_argument('--val_dir', help='Val directory')
    parser.add_argument('--checkpoint_dir', help='Val  directory')
    parser.add_argument('--experiment_name', help='exp name')
    parser.add_argument('--is_distributed', type=bool,
                        default=False, help="DDP enabled")
    # parser.add_argument('--world_size',help='World size') # Enable for ddp
    parser.add_argument('--local_rank', type=int, default=-1,
                        metavar='N', help='Local process rank.')
    parser.add_argument('--question_dir', help='Q directory')
    parser.add_argument('--ngpus',type=int,default=0, help='Q directory')
    parser.add_argument('--machine_type',default='single',help='dp or ddp or single or cpu')
    parser.add_argument('--wandb_status',default='online', help='wandb set to online for online sync else disabled')
    parser.add_argument('--image_size',type=int,default=384,help=' image size height and width')

    args = parser.parse_args()

    EPOCHS = args.num_epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    LEARNING_RATE = args.learning_rate
    CHECKPOINT_DIR = args.checkpoint_dir
    EXPERIMENT_NAME = args.experiment_name
    LOCAL_RANK = args.local_rank
    Q_PATH = args.question_dir
    NUM_GPUS = args.ngpus
    WANDB_STATUS = args.wandb_status
    IMAGE_SIZE = args.image_size
    MACHINE_TYPE=args.machine_type
    ic(MACHINE_TYPE)


    ic(NUM_GPUS)
    device_ids = []
    for i in range(NUM_GPUS):
        device_ids.append(i)


    # Record start time
    start_time = time.time()
    log_file = os.path.join(CHECKPOINT_DIR,'statistics.log')
    ic(log_file)
    
    if not os.path.exists(log_file):
        open(log_file, 'w+').close()
        print("log file created")


    logging.basicConfig(filename=log_file,level=logging.INFO,filemode="w") # #,
    logger = logging.getLogger("mylogger")

    ngpus = args.ngpus

    ic(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)

    wandb.init(project='CircuitBLIP',
            config={
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'train_batch_size': TRAIN_BATCH_SIZE,
                'val_batch_size': VAL_BATCH_SIZE
            }
                ,mode=WANDB_STATUS)
    wandb.run.name = EXPERIMENT_NAME

    ic(args.is_distributed)

    # model = BlipForQuestionAnswering.from_pretrained(
    #     "Salesforce/blip-vqa-base")
    #model.config.to_json_file("config.json")

    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",ignore_mismatched_sizes=True) # ,config=config_file,ignore_mismatched_sizes=True)
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
   
    print(model.config)
    torch.cuda.empty_cache()

    rank = -1

    train_dataset = VQACircuitDataset(TRAIN_DIR, Q_PATH, split='train')
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, collate_fn=vqa_collate_fn,pin_memory=False)

    val_dataset = VQACircuitDataset(VAL_DIR, Q_PATH, split='val')
    val_dataloader = DataLoader(
        val_dataset, shuffle=True, batch_size=VAL_BATCH_SIZE, collate_fn=vqa_collate_fn,pin_memory=False)

    ic("data loaded")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ic(device)
    loss_function = torch.nn.CrossEntropyLoss()

    if MACHINE_TYPE=='dp' and ngpus>1:
        model = nn.DataParallel(model,device_ids=device_ids)
        ic(device_ids)

    model.to(device)
    ic(next(model.parameters()).is_cuda)
    model.train()
    # ic(torch.cuda.current_device(), torch.cuda.device_count())

    best_accuracy = [0, 0.0]

    for epoch in range(EPOCHS):
        start_time_epoch = time.time()
        ic(epoch)
        train_loss = 0.0
        train_total = 0
        val_loss = 0.0
        total = 0
        accuracy_batch = 0.0
        accuracy_train_batch = 0.0
        num_batches = 0

        for idx, (images, questions, answers,qtype) in enumerate(tqdm(train_dataloader)):
            train_total += len(answers)
            num_batches += 1
            #ic(idx, images, questions, answers)
            inputs = processor(images=images, text=questions,
                               return_tensors="pt", padding=True)

            if IMAGE_SIZE!=384:
                #ic(inputs['pixel_values'].shape)
                im_processor =  BlipImageProcessor(image_size=IMAGE_SIZE)
                inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
                inputs['pixel_values'] = inputs_im['pixel_values']
                #ic(inputs['pixel_values'].shape)
            
            labels = processor(
                text=answers, return_tensors="pt", padding=True).input_ids
            # print(inputs,labels)
            inputs["labels"] = labels

            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(**inputs)

            loss = outputs.loss

            #ic(train_loss)
            if MACHINE_TYPE == 'dp' and ngpus>1:
                #ic(loss,MACHINE_TYPE)
                train_loss += loss.sum().item()
                preds = model.module.generate(**inputs)
            else:
                #ic(loss,MACHINE_TYPE)
                train_loss += loss.item()
                preds = model.generate(**inputs)
    
            predicted = processor.batch_decode(preds, skip_special_tokens=True)
            #ic(predicted)
            accuracy_train_batch += len([i for i,
                                        j in zip(predicted, answers) if i == j])
            ic(loss.sum().item())#,num_batches,predicted,answers,questions,epoch,'train')

            if MACHINE_TYPE == 'dp' and ngpus>1:
                loss.sum().backward()
            else:
                loss.backward()

            optimizer.step()

            del outputs,loss,inputs,labels
            torch.cuda.empty_cache()
            gc.collect()

        #ic(epoch,num_batches,train_loss)
        train_loss = train_loss / (num_batches*NUM_GPUS)
        #wandb.log({'epoch': epoch+1, 'train loss': train_loss})

        train_accuracy = (100 * accuracy_train_batch / train_total)

        logger.info(f'Epoch {epoch} Train loss : {train_loss} Train accuracy : {train_accuracy}')

        # Eval loop
        model.eval()
        num_batches = 0

        with torch.no_grad():
            for idx, (images, questions, answers,qtype) in enumerate(tqdm(val_dataloader)):
                #ic(epoch)
                num_batches += 1
                #ic(idx, images, questions, answers)
                total += len(answers)
                inputs = processor(images=images, text=questions,
                               return_tensors="pt", padding=True)

                if IMAGE_SIZE!=384:
                    inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
                    inputs['pixel_values'] = inputs_im['pixel_values']
                    #ic(inputs['pixel_values'].shape)

                labels = processor(
                    text=answers, return_tensors="pt", padding=True).input_ids
                inputs["labels"] = labels
                inputs = inputs.to(device)
                labels = labels.to(device)


                outputs = model(**inputs)
                outputs = outputs
                loss = outputs.loss            

                if MACHINE_TYPE == 'dp' and ngpus>1:
                    #ic(loss,MACHINE_TYPE)
                    val_loss += loss.sum().item()
                    preds = model.module.generate(**inputs)
                else:
                    #ic(loss,MACHINE_TYPE)
                    val_loss += loss.item()
                    preds = model.generate(**inputs)

                predicted = processor.batch_decode(
                    preds, skip_special_tokens=True)
                
                ic(loss.sum().item()) #,num_batches,predicted,answers,epoch,'val')

                accuracy_batch += len([i for i,
                                      j in zip(predicted, answers) if i == j])

                del outputs,loss,inputs,labels
                gc.collect()


        #ic(epoch,num_batches,train_loss)
        val_loss = val_loss / (num_batches*NUM_GPUS)

        val_accuracy = (100 * accuracy_batch / total)
       
        logger.info(f'Epoch {epoch} Val loss : {val_loss} Val accuracy : {val_accuracy}')

        wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'epoch': epoch+1,
                'offline': True
            })   


        # Save model per epoch
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        
        if epoch == 0:
            torch.save(save_obj, os.path.join(
                CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))

        if val_accuracy > best_accuracy[1] and epoch > 0:
            os.remove(os.path.join(CHECKPOINT_DIR,
                      'checkpoint_%02d.pth' % best_accuracy[0]))
            ic("old checkpoint removed")
            best_accuracy[0] = epoch
            best_accuracy[1] = val_accuracy
            torch.save(save_obj, os.path.join(
                CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))

        # Record end time
        end_time_epoch = time.time()
        run_time_epoch = end_time_epoch - start_time_epoch
        logger.info(f'Epoch {epoch} run time: {run_time_epoch:.2f} seconds')
        
        

    # Record end time
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f'Script run time: {run_time:.2f} seconds')