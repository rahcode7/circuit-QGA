
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering,BlipImageProcessor
from datasets import VQACircuitDataset,vqa_collate_fn
from torch.utils.data import DataLoader
import torch
import wandb
from tqdm import tqdm
import os 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group
import argparse
from icecream import ic 
from datetime import timedelta
import torch.multiprocessing as mp
import logging
import time 
from utils.helpers import set_seed
from accelerate import Accelerator
from icecream import ic 
from transformers import Adafactor


def average_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= dist.get_world_size()


def train(model,train_dataloader,processor,accelerator,device,optimizer,IMAGE_SIZE,ACCUMULATION_STEPS):
    model.train()

    train_losses = []
    train_corrects = []

    for idx, (images,questions,answers,qtype) in enumerate(tqdm(train_dataloader)):
        with accelerator.accumulate(model):
            inputs = processor(images=images, text=questions,return_tensors="pt", padding=True)
            if IMAGE_SIZE!=384:
                    #ic(inputs['pixel_values'].shape)
                    im_processor =  BlipImageProcessor()
                    inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
                    inputs['pixel_values'] = inputs_im['pixel_values']

            labels = processor(text=answers, return_tensors="pt",padding=True).input_ids
            inputs["labels"] = labels

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.module(**inputs)
            loss = outputs.loss
            
            train_losses.append(accelerator.gather(outputs.loss))
            

            preds = model.module.generate(**inputs)
            predicted = processor.batch_decode(preds, skip_special_tokens=True)
            train_batch_corrects = len([i for i,
                                            j in zip(predicted, answers) if i == j])
        
            train_batch_corrects = torch.Tensor(train_batch_corrects).to(device)
            ic(outputs.loss,train_batch_corrects,train_batch_corrects)
            train_corrects.append(accelerator.gather(train_batch_corrects))
            ic(train_corrects)

            #train_corrects.append(train_batch_corrects)
            # train_corrects.append(accelerator.gather_for_metrics(torch.tensor([train_batch_corrects])))


            accelerator.backward(loss)
            # Gradient accumulation 
            if (idx + 1)% ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

    ic(device,train_losses,len(train_dataloader),len(train_dataloader.dataset))
    train_loss = torch.sum(train_losses) / len(train_dataloader)
    train_accuracy = torch.sum(train_corrects) / len(train_dataloader.dataset)

    return train_loss,train_accuracy

def evaluate(model,val_dataloader,processor,accelerator,device,IMAGE_SIZE):
    val_losses = []
    val_corrects = []
    if accelerator.is_main_process:
        with torch.no_grad(): 
            model.eval()
            num_batches_val = 0
            for idx,(images,questions,answers,qtype) in enumerate(tqdm(val_dataloader)):
                inputs = processor(images=images, text=questions, return_tensors="pt",padding=True)
                labels = processor(text=answers, return_tensors="pt",padding=True).input_ids
                inputs["labels"] = labels

                if IMAGE_SIZE!=384:
                    im_processor =  BlipImageProcessor()
                    inputs_im = im_processor(images=images,size={"height":IMAGE_SIZE,"width":IMAGE_SIZE},return_tensors="pt")
                    inputs['pixel_values'] = inputs_im['pixel_values']

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model.module(**inputs)
                val_loss.append(accelerator.gather(outputs.loss))

                preds  = model.module.generate(**inputs)

                predicted = processor.batch_decode(preds, skip_special_tokens=True)

                val_batch_corrects  = len([i for i, j in zip(predicted,answers) if i == j])

                val_batch_corrects = torch.Tensor(val_batch_corrects).to(device)
                val_corrects.append(accelerator.gather(val_batch_corrects))
                ic(val_corrects)
        
    ic(device,val_losses,len(val_dataloader),len(val_dataloader.dataset))
    val_loss = torch.sum(val_losses) / len(val_dataloader)
    val_accuracy = torch.sum(val_corrects) / len(val_dataloader.dataset)

    return val_loss,val_accuracy


def run_ddp_accelerate(args):
   
    EPOCHS = args.num_epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    #TEST_BATCH_SIZE = args.test_batch_size
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    LEARNING_RATE = args.learning_rate
    CHECKPOINT_DIR  = args.checkpoint_dir
    EXPERIMENT_NAME = args.experiment_name
    #LOCAL_RANK = args.local_rank
    Q_PATH = args.question_dir
    ACCUMULATION_STEPS = 4
    WANDB_STATUS = args.wandb_status
    IMAGE_SIZE = args.image_size

    accelerator = Accelerator(gradient_accumulation_steps=ACCUMULATION_STEPS)
    ic("setting up done ")

    # Record start time
    start_time = time.time()
    log_file = os.path.join(CHECKPOINT_DIR,'statistics.log')

    if not os.path.exists(log_file):
        open(log_file, 'w+').close()
        print("log file created")

    ic(log_file)
    logging.basicConfig(filename=log_file,level=logging.INFO,filemode="w") # #,
    logger = logging.getLogger("mylogger")


    ic(TRAIN_BATCH_SIZE,VAL_BATCH_SIZE)

    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base",ignore_mismatched_sizes=True)
    model.gradient_checkpointing_enable()
    processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    
    ic("model loaded")
    
    if accelerator.is_main_process:
        wandb.init(project='CircuitBLIP',
                config = {
                'learning_rate':LEARNING_RATE,
                'epochs':EPOCHS,
                'train_batch_size': TRAIN_BATCH_SIZE,
                'val_batch_size':VAL_BATCH_SIZE
                },mode=WANDB_STATUS)

    if accelerator.is_main_process:
        wandb.watch(model)


    train_dataset = VQACircuitDataset(TRAIN_DIR,Q_PATH,split='train')
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=TRAIN_BATCH_SIZE,
                                 collate_fn=vqa_collate_fn,
                                 shuffle=False,
                                 sampler = DistributedSampler(train_dataset,drop_last=False))


    val_dataset = VQACircuitDataset(VAL_DIR,Q_PATH,split='val')
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=VAL_BATCH_SIZE,
                                collate_fn=vqa_collate_fn,
                                shuffle=False,
                                sampler = DistributedSampler(val_dataset,drop_last=False))

    print("data loaded")
    device = accelerator.device
    ic(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    #optimizer = Adafactor(model.parameters(),lr=LEARNING_RATE,weight_decay=0.05,relative_step=False)


    train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader,val_dataloader, model, optimizer
    )

    
    for epoch in range(EPOCHS):

        train_loss,train_accuracy = train(model,train_dataloader,processor,accelerator,device,optimizer,IMAGE_SIZE,ACCUMULATION_STEPS)
        val_loss, val_accuracy = evaluate(model,val_dataloader,processor,accelerator,device,optimizer,IMAGE_SIZE)
             
        if accelerator.is_main_process:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'epoch': epoch+1
            })      

        #Save model per epoch
    #     save_obj = {
    #                 'model': model.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 #'config': model.config,
    #                 'epoch': epoch,
    #             } 

    #     if epoch==0 and rank==0:
    #         torch.save(save_obj, os.path.join(CHECKPOINT_DIR, 'checkpoint_%02d.pth'% epoch))

    #     if rank==0 and val_accuracy > best_accuracy[1] and epoch>0:
    #             os.remove(os.path.join(CHECKPOINT_DIR, 'checkpoint_%02d.pth'%best_accuracy[0]))
    #             ic("old checkpoint removed")
    #             best_accuracy[0] = epoch
    #             best_accuracy[1] = val_accuracy
    #             torch.save(save_obj, os.path.join(CHECKPOINT_DIR, 'checkpoint_%02d.pth'% epoch))
    #cleanup()
    #wandb.finish() 


    wandb.finish()
    ic("Completed training")

    # Record end time
    end_time = time.time()
    # Calculate script run time
    run_time = end_time - start_time
    # Log the script run time
    logger.info(f'Script run time: {run_time:.2f} seconds')


if __name__ == "__main__":

    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=5, type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=2e-5 , type=int, help='Learning rate')
    parser.add_argument('--train_batch_size', default=8, type=int, help='train batch size')
    parser.add_argument('--val_batch_size', default=8, type=int, help='val batch size')
    parser.add_argument('--test_batch_size', default=8, type=int, help='test batch size')
    parser.add_argument('--train_dir',help='train directory')
    parser.add_argument('--val_dir',help='Val directory')
    parser.add_argument('--checkpoint_dir',help='Val  directory')
    parser.add_argument('--experiment_name',help='exp name')
    #parser.add_argument('--world_size',help='World size') # Enable for ddp
    parser.add_argument('--is_distributed',default=False,type=bool,help="Is distributed")
    #parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='Local process rank.')
    parser.add_argument('--question_dir', help='Q directory')
    parser.add_argument('--ngpus',type=int, help='Q directory')
    parser.add_argument('--machine_type',default='single',help='dp or ddp or single or cpu')
    parser.add_argument('--wandb_status', help='wandb set to online for online sync else disabled')
    parser.add_argument('--image_size',type=int,default=384,help=' image size height and width')


    args = parser.parse_args()    

    run_ddp_accelerate(args)
    