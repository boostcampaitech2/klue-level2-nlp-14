# run_prompt.py
"""
Execution file for PTR(Prompt Tuning with Rules for text classification) method.
Functions:
    main(command_args): Training, inference are conducted according to command args.
"""
import os
from tqdm import tqdm, trange
from collections import Counter

import numpy as np

import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

import wandb

from arguments import get_args_parser
from templating import get_temps
from modeling import get_model, get_tokenizer
from data_prompt import REPromptDataset
from optimizing import get_optimizer
from ....solution.utils import FocalLoss, get_confusion_matrix, set_seeds

# Helper Functions
def f1_score_for_PTR(output, label, rel_num, na_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]

        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)

    return micro_f1

def evaluate(model, dataset, dataloader, is_test=False):
    model.eval()
    scores = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model(**batch)
            res = []
            for i in dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]                
                _res = _res.detach().cpu()
                res.append(_res)
            logits = torch.stack(res, 0).transpose(1,0)
            labels = batch['labels'].detach().cpu().tolist()
            all_labels+=labels
            scores.append(logits.cpu().detach())
        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy() 
        all_labels = np.array(all_labels)

        # Save scores and labels
        np.save("scores.npy", scores)
        np.save("all_labels.npy", all_labels)

        if not is_test:
            # Compute confusion matrix
            cm_fig = get_confusion_matrix(scores, np.array(all_labels))
            wandb.log({'confusion matrix': wandb.Image(cm_fig)})

            pred = np.argmax(scores, axis = -1)
            mi_f1 = f1_score_for_PTR(pred, all_labels, dataset.num_class, dataset.NA_NUM)
        else:
            mi_f1 = None

    return mi_f1

def main(args):
    # Set-up WANDB
    os.environ["WANDB_PROJECT"] =args.wandb_project
    wandb.init(config=args, entity='kiyoung2', name=args.wandb_run_name)
    
    # Set seed
    set_seeds(args.seed)

    # Load tokenizer
    tokenizer = get_tokenizer(special=[])

    # Load Prompt Template
    temps = get_temps(tokenizer)

    # Get training data
    # If the dataset has been saved, 
    # the code ''dataset = REPromptDataset(...)'' is not necessary.
    for split in ['train, val, test']:
        dataset = REPromptDataset(
            path  = args.data_dir, 
            name = f'{split}.txt', 
            rel2id = args.data_dir + "/" + "rel2id.json", 
            temps = temps,
            tokenizer = tokenizer,)
        dataset.save(path = args.output_dir, name = split)

    train_dataset, val_dataset, test_dataset = (REPromptDataset.load(
        path = args.output_dir, 
        name = split, 
        temps = temps,
        tokenizer = tokenizer,
        rel2id = args.data_dir + "/" + "rel2id.json") 
        for split in ['train','val','test'])

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset.cuda()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    val_dataset.cuda()
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size)

    test_dataset.cuda()
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size)

    # Get model
    model = get_model(tokenizer, train_dataset.prompt_label_idx)
    wandb.watch(model, log_freq=1000)

    # Set Optimizer & Scheduler & Loss function
    optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
    criterion = FocalLoss(gamma=0.5)

    # Training
    mx_res = 0.0
    hist_mi_f1 = []
    mx_epoch = None

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        model.zero_grad()
        tr_loss = 0.0
        global_step = 0 
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            logits = model(**batch)
            labels = train_dataset.prompt_id_2_label[batch['labels']] # (Batch, N_MASK=5)
            
            loss = 0.0
            
            for index, i in enumerate(logits):
                # i : (Batch, N_MLM_Head_label=(N_subj_entity_type, N_Relation_prompt_tokens, N_obj_entity_type))
                loss += criterion(i, labels[:,index])
            loss /= len(logits)

            # Relation Label Loss
            res = []
            for i in train_dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]
                res.append(_res)
            final_logits = torch.stack(res, 0).transpose(1,0) # [Batch, N_Relation_Label]

            loss += criterion(final_logits, batch['labels'])

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer_new_token.step()
                scheduler_new_token.step()
                model.zero_grad()
                global_step += 1
                print (tr_loss/global_step, mx_res)

        mi_f1 = evaluate(model, val_dataset, val_dataloader)
        hist_mi_f1.append(mi_f1)
        
        wandb.log({'train/loss':tr_loss/global_step})
        wandb.log({'eval/micro_f1':mi_f1})
        
        # Save Best Checkpoint
        if mi_f1 > mx_res:
            mx_res = mi_f1
            mx_epoch = epoch
            torch.save(model.state_dict(), args.output_dir+"/"+'parameter'+str(epoch)+".pkl")

    print(hist_mi_f1)

    # Predict Test data for submission
    model.load_state_dict(torch.load(args.output_dir+"/"+'parameter'+str(mx_epoch)+".pkl"))
    mi_f1, _ = evaluate(model, test_dataset, test_dataloader, is_test=True)

    print(mi_f1)

if __name__ == "__main__":
    args = get_args_parser()
    main(args)