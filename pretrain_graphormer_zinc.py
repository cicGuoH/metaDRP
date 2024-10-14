# -*- coding: utf-8 -*-

import pickle, os
import copy
import pandas as pd
import numpy as np
import random
from tqdm import tqdm 
import torch
import torch.nn as nn
import datetime
from torch.utils.data import DataLoader
from model_utils.lr_schedular import build_scheduler
from models.meta_model_architectures import GraphLevelGraphormer
from data_utils.dataset import PyGZINCDataset, BatchedDrugDataset
from data_utils.collator import collator_forDrug
from torch.utils.tensorboard import SummaryWriter

rnd_seed = 42


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]

#%% pretrain
def build_task_folder(task_name):
    par_path = os.path.abspath(os.path.dirname(os.getcwd()))
    task_path = os.path.join(par_path, task_name)
    saved_model_path = os.path.join(task_path, "saved_model")
    log_path = os.path.join(task_path, "training_log")
    visual_path = os.path.join(task_path, "visual_output")
    
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
        
    outputs = (saved_model_path, log_path, visual_path)
    outputs = (os.path.abspath(item) for item in outputs)
    return outputs

def pretrain_drugenc(args):
    
     
    train_set = PyGZINCDataset(split="train")
    test_set = PyGZINCDataset(split="val")

    dataset_train = BatchedDrugDataset(train_set, args.drug_max_nodes, args.drug_multi_hop_max_dist, args.spatial_pos_max)
    dataset_test = BatchedDrugDataset(test_set, args.drug_max_nodes, args.drug_multi_hop_max_dist, args.spatial_pos_max)
    
    train_dataloader = DataLoader(dataset_train, collate_fn=collator_forDrug, batch_size=args.num_batches_per_epoch)
    test_dataloader = DataLoader(dataset_test, collate_fn=collator_forDrug, batch_size=args.num_batches_per_epoch)
    
    
    saved_model_path, log_path, visual_path = build_task_folder(args.task_name)
    model = GraphLevelGraphormer(args=args, output_dim=1)
    model.to(args.device)
    params = torch.nn.utils.parameters_to_vector(parameters=model.parameters())
    print('Number of parameters of the base network = {0:,}.\n'.format(params.numel()))
   
    # logging
    tb_writer = SummaryWriter(
        log_dir=log_path,
        purge_step=args.resume_epoch if args.resume_epoch > 0 else None
    )        
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.base_lr, 
                                  weight_decay=args.weight_decay, 
                                  )
    
    scheduler = build_scheduler(args, optimizer, len(train_dataloader))
    
    
    if args.resume_epoch > 0:
        ckpt_path = os.path.join(saved_model_path, 'Epoch_{0:d}.pt'.format(args.resume_epoch))
        saved_ckpt = torch.load(
            f=ckpt_path,
            map_location=args.device)
        model.graph_encoder.load_state_dict(saved_ckpt["graphencoder"])
        model.fc.load_state_dict(saved_ckpt["mlp"])
        optimizer.load_state_dict(saved_ckpt["opt_state_dict"])
        scheduler.load_state_dict(saved_ckpt["sche_state_dict"])
    
    loss_function = nn.L1Loss(reduction="mean")
    best_loss = 9999.
    loss_monitor = {}
    for epoch_id in range(args.resume_epoch, args.resume_epoch+args.total_epochs):
        model.train()
        loss_monitor["Train_loss_epoch"] = 0.
        loss_monitor["Val_loss_epoch"] = 0.
        for iter_count, batch in enumerate(tqdm(train_dataloader)):
            
            current_iter =  epoch_id * len(train_dataloader) + iter_count + 1
            
            batch.to(args.device)
            y = batch.y
            output = model(batch).view((-1,))
            loss = loss_function(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step_update(current_iter)
            loss_monitor["Train_loss_epoch"] += loss.item()
            
        tb_writer.add_scalar(tag="learning_rate", scalar_value=optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch_id+1)
        tb_writer.add_scalar(tag="Train_loss_epoch", scalar_value=loss_monitor["Train_loss_epoch"]/len(train_dataloader), global_step=epoch_id+1)
        
        model.eval()
        for batch in test_dataloader:
            batch.to(args.device)
            y = batch.y
            with torch.no_grad():
                output = model(batch).view((-1,))
                val_loss = loss_function(output, y)
            loss_monitor["Val_loss_epoch"] += val_loss.item()
        del val_loss
        del output
        loss_epoch = loss_monitor["Val_loss_epoch"]/len(test_dataloader)
        tb_writer.add_scalar(tag="Val_loss_epoch", scalar_value=loss_epoch, global_step=epoch_id+1)
        
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            best_ckpt = {
                "graphencoder": model.graph_encoder.state_dict(),
                "mlp":model.fc.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "sche_state_dict":scheduler.state_dict()
                }
            best_ckpt_path = os.path.join(saved_model_path, 'best_model.pt')
            torch.save(obj=best_ckpt, f=best_ckpt_path)
        # save model per epoch    
        ckpt = {
            "graphencoder": model.graph_encoder.state_dict(),
            "mlp":model.fc.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "sche_state_dict":scheduler.state_dict()
            }
        ckpt_path = os.path.join(saved_model_path, 'Epoch_{0:d}.pt'.format(epoch_id + 1))
        torch.save(obj=ckpt, f=ckpt_path)
        print('State dictionaries are saved into {0:s}\n'.format(ckpt_path))
    print('\nClose tensorboard summary writer')
    tb_writer.close()
            
    
if __name__ == "__main__":
    
    
    args = {
        "drug_max_nodes":64,
        "spatial_pos_max":64, 
        "drug_num_atoms":64*9,
        "drug_num_in_degree":64,
        "drug_num_out_degree":64,
        "drug_num_edges":64*3,
        "drug_num_spatial":64,
        "drug_num_edge_dist": 5,
        "drug_multi_hop_max_dist":5,
        "drug_n_encode_layers":12,
        "drug_embed_dim":80,
        "drug_num_heads":8,
        "layer_dropout_rate":0.0,
        
        # model shared args
        "embed_dim":80,
        "dropout_rate":0.1,
        "q_noise":0.0,
        "qn_block_size":8,
        
        # training
        "task_name":"pretrain_DrugEncoder_slim_zinc_%s" % (datetime.datetime.now().strftime('%m-%d')),
        "device":"cuda", 
        "base_lr":2e-4, 
        "resume_epoch":0, 
        "total_epochs":10000, 
        "train_flag":True,
        "num_batches_per_epoch":64,
        "warmup_lr":5e-7,
        "lr_scheduler_name":"cosine", 
        "decay_rate":0.1,
        "min_lr":1e-9,
        "warmup_epochs":600, 
        "decay_epochs":30, 
        "weight_decay":0.05, 
        "clip_grad":5.0,
        }
    
    args = DotDict(args)    
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    pretrain_drugenc(args)