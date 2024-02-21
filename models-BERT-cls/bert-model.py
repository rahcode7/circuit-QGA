from transformers import BertTokenizer,BertForSequenceClassification, AdamW, BertConfig,get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
import pandas as pd 
import wandb
import torch
import time
import datetime
import random
import logging
import numpy as np
from  utils import BERTtokenization,flat_accuracy,format_time,load_ckp,assign_device
import argparse
from icecream import ic 
import os 
import json 
from pickle import dump,load

if __name__ == "__main__":
    ### Assign GPU or CPU 
    #device = assign_device()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ic(device)

    #set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_dir', help='Q directory')
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
    parser.add_argument('--checkpoint_dir', help='Val  directory')
    parser.add_argument('--experiment_name', help='exp name')
    # parser.add_argument('--world_size',help='World size') # Enable for ddp
    parser.add_argument('--local_rank', type=int, default=-1,
                        metavar='N', help='Local process rank.')
    parser.add_argument('--ngpus',type=int,default=0, help='Q directory')
    parser.add_argument('--machine_type',default='single',help='dp or ddp or single or cpu')
    parser.add_argument('--wandb_status',default='online', help='wandb set to online for online sync else disabled')

    args = parser.parse_args()

    EPOCHS = args.num_epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    LEARNING_RATE = args.learning_rate
    CHECKPOINT_DIR = args.checkpoint_dir
    EXPERIMENT_NAME = args.experiment_name
    Q_PATH = args.question_dir
    NUM_GPUS = args.ngpus
    WANDB_STATUS = args.wandb_status
    MACHINE_TYPE=args.machine_type

    args = parser.parse_args()

    OUTPUT_FOLDER = "datasets/results/bert/"


    # Record start time
    start_time = time.time()
    log_file = os.path.join(CHECKPOINT_DIR,'statistics.log')
    ic(log_file)
    
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
        print("log file created")
    else:
        print("log file exists")


    logging.basicConfig(filename=log_file,level=logging.INFO,filemode="a") # #,
    logger = logging.getLogger("mylogger")


    # Wandb 
    wandb.init(project='CircuitBERT',
            config={
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'train_batch_size': TRAIN_BATCH_SIZE,
                'val_batch_size': VAL_BATCH_SIZE
            }
                ,mode=WANDB_STATUS)
    wandb.run.name = EXPERIMENT_NAME

    ### Read questions type datasets
    #Q_PATH = "/Users/rahulmehta/Desktop/MSIIIT/QGen-circuits/circuit-QGA/datasets/questions/all/master_sample_100.json"
    df = pd.read_json(Q_PATH)
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['qtype'])

    dump(LE, open(OUTPUT_FOLDER + '/label_encoder.pkl', 'wb'))
    LE = load(open(OUTPUT_FOLDER + '/label_encoder.pkl', 'rb'))
    

    label_classes = list(LE.classes_)   
    ic(label_classes)
    with open(OUTPUT_FOLDER +  "label_classes.json", 'w') as f:
        # indent=2 is not needed but makes the file human-readable 
        # if the data is nested
        json.dump(label_classes, f, indent=2) 

    ic(df.head(5))
    #ic(df['label'].unique())

    train_df = df[df.splittype=='train']
    sentences = train_df.question.values
    labels = train_df.label.values
    #ic(labels)


    ### Tokenize
    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


    input_ids,attention_masks = BERTtokenization(tokenizer,sentences)
    labels = torch.tensor(labels)


    ### Train dataset
    train_dataset = TensorDataset(input_ids,attention_masks,labels) 


    ### Test dataaet prepare
    val_df = df[df.splittype=='train']
    sentences = val_df.question.values
    labels = val_df.label.values

    input_ids,attention_masks = BERTtokenization(tokenizer,sentences)
    labels = torch.tensor(labels)

    ### Train dataset
    val_dataset = TensorDataset(input_ids,attention_masks,labels) 


    ### Dataloaders
    #batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = TRAIN_BATCH_SIZE # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = VAL_BATCH_SIZE # Evaluate with this batch size.
        )
    

    ########### Modelling 
    

    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 5, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        return_dict=False 
    )

    # Tell pytorch to run this model on the GPU.
    ic(device)
    #if device == "cuda":

    model.to(device)


    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = EPOCHS

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    

  
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    best_accuracy = [0, 0.0]

    # Restart from latest checkpoint if available 

    starting_name = 'checkpoint'
    stats_name = 'statistics'

    file_list = os.listdir(CHECKPOINT_DIR)
    ic(file_list)
    START = 0

    if file_list:
        checkpoint_files = [filename for filename in file_list if filename.startswith(starting_name)]
        ic(checkpoint_files)
        # Get latest checkpoint
        if checkpoint_files:
            checkpoint_files.sort()
            LATEST_CHECKPOINT=checkpoint_files[-1]
            START = int(checkpoint_files[-1].split(".pth")[0][-1:])+1
            ic("Previous checkpoint found",LATEST_CHECKPOINT,START)
        
     
    # Check log file for best validation accuracy

    prev_accuracy = []
    prev_epoch = []
    stats = os.path.join(CHECKPOINT_DIR,'statistics.log')

    if os.path.exists(stats):
        with open(stats) as f:
            lines = f.readlines()
            #ic(lines)
            for line in lines:
                if 'Val accuracy' in line:
                    eid =line.index('Epoch') + 6
                    aid =line.index('Val accuracy') + 15

                    prev_epoch.append(int(line[eid]))
                    prev_accuracy.append(float(line[aid:aid+6]))
                    #ic(prev_accuracy,prev_epoch)
    ic(prev_accuracy,prev_epoch)

    if prev_accuracy:
        max_index, max_val = max(enumerate(prev_accuracy),key=lambda x:x[1])
        ic(max_val,max_index)
        best_accuracy = [prev_epoch[max_index],max_val]
        START = prev_epoch[-1] +1
        ic("Previous best accuracy found",best_accuracy[1],"best epoch",best_accuracy[0])
        chk_file = os.path.join(CHECKPOINT_DIR,'checkpoint_0' + str(max_index) + '.pth')
        ic(chk_file)
        model, optimizer,_= load_ckp(chk_file, model, optimizer)
    else:
        # Remove any checkpoints 
        for fname in os.listdir(CHECKPOINT_DIR):
            if fname.startswith("checkpoint"):
                os.remove(os.path.join(CHECKPOINT_DIR,fname))
                ic("removed checkpoint",fname)
            START=0

    ic("Running from epoch",START,"to epoch",EPOCHS)

    # For each epoch...
    for epoch in range(START,EPOCHS):
        start_time_epoch = time.time()
        ic(epoch)

        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            # ic(b_input_ids,b_labels,b_input_mask)

            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            #ic(loss)
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(train_loss))
        print("  Training epcoh took: {:}".format(training_time))
            
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # Report the final accuracy for this validation run.
        val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(val_accuracy))

        # Calculate the average loss over all of the batches.
        val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': train_loss,
                'Valid. Loss': val_loss,
                'Valid. Accur.': val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        wandb.log({'epoch': epoch+1, 'train loss': train_loss})
        #wandb.log({'epoch': epoch+1, 'train accuracy': train_accuracy})
        wandb.log({'epoch': epoch+1, 'val loss': val_loss})
        wandb.log({'epoch': epoch+1, 'val accuracy': val_accuracy}) 

        logger.info(f'Epoch {epoch} Train loss : {train_loss} Val accuracy : {val_accuracy}')
        logger.info(f'Epoch {epoch} Val loss : {val_loss} Val accuracy : {val_accuracy}')


        # Save model per epoch
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        
        if val_accuracy == 1.0:
            best_accuracy[0] = epoch
            best_accuracy[1] = float(val_accuracy)
            torch.save(save_obj, os.path.join(
                CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
            break
        
        if epoch == 0:   
            ic("Saving 0 epoch checkpoint")
            torch.save(save_obj, os.path.join(
                CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
            best_accuracy[0] = epoch
            best_accuracy[1] = float(val_accuracy)

        elif val_accuracy > best_accuracy[1] and epoch > 0:
            os.remove(os.path.join(CHECKPOINT_DIR,
                    'checkpoint_%02d.pth' % best_accuracy[0]))
            ic("old checkpoint removed")
            best_accuracy[0] = epoch
            best_accuracy[1] = float(val_accuracy)
            torch.save(save_obj, os.path.join(
                CHECKPOINT_DIR, 'checkpoint_%02d.pth' % epoch))
        else:
            continue

        # Record end time
        end_time_epoch = time.time()
        run_time_epoch = end_time_epoch - start_time_epoch
        logger.info(f'Epoch {epoch} run time: {run_time_epoch:.2f} seconds')
        
        

    # Record end time
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f'Script run time: {run_time:.2f} seconds')


    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    
    # Display floats with two decimal places.
    #pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    ic(df_stats)     


    df_stats.to_csv("datasets/results/bert/training_stats.csv",index=False)


