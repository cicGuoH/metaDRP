# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:53:22 2024

@author: GUI
"""

from traditional_train_system import TraditionalBase
import os.path as osp
import numpy as np
import pandas as pd
import random
import torch
import argparse
from data_utils.dataset import DRDataloderForTraditionalTraining, DRDataloderForFewShotLearning

_random_seed = 4132231

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
        
def arg_parse():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--drug_dataset_name', type=str, default="GDSC1", help="directory path of drug")
    parser.add_argument('--ccl_dataset_name', type=str, default="CCL", help="directory path of ccl")
    parser.add_argument('--omics_combination', type=str, default="E", help="omics combination, etc. ECDP")
    parser.add_argument('--drug_pretrain_path', type=str, default="../model_logging/pretrain_DrugEncoder_slim_zinc_12-10", help="file path of drug pretrain models")
    parser.add_argument('--sample_name', type=str, default="DepMap_ID")
    parser.add_argument('--measurement_method', type=str, default="IC50")
    
    parser.add_argument('--dataset_name_train', type=str, default="maml_train")
    parser.add_argument('--dataset_name_val', type=str, default="maml_val")
    parser.add_argument('--rm_seed', type=int, default=31415, help="random seed of sampling train dataset")
    parser.add_argument('--val_seed', type=int, default=0, help="random seed of sampling val/test dataset")
    parser.add_argument('--k', type=int, default=10, help="number of support ccls per task")
    parser.add_argument('--val_samples_per_batch', type=int, default=64, help="number of query ccls per task")
    parser.add_argument('--num_batches_per_iter', type=int, default=128, help="number of tasks per iter")
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--mini_batch', action='store_false', help="whether to use mini-batch way to evaluate on a drug-tissue dataset, suitable when size is big")
    
    # train
    parser.add_argument('--model_name', type=str, default="metaDRP", help="model name, etc. metaDRP")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--total_epochs', type=int, default=500)
    parser.add_argument('--pretrain_flag', type=bool, default=True)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--first_order', type=bool, default=True)

    # lr scheduler
    parser.add_argument('--lr_scheduler_name', type=str, default="cosine")
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--warmup_lr', type=float, default=8e-5)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--decay_epochs', type=int, default=5)
    parser.add_argument('--clip_grad', type=float, default=0.)
    
    ## model configs
    # drug_enc
    parser.add_argument('--drug_max_nodes', type=int, default=64)
    parser.add_argument('--spatial_pos_max', type=int, default=64)
    parser.add_argument('--drug_num_atoms', type=int, default=64*9)
    parser.add_argument('--drug_num_in_degree', type=int, default=64)
    parser.add_argument('--drug_num_out_degree', type=int, default=64)
    parser.add_argument('--drug_num_edges', type=int, default=64*3)
    parser.add_argument('--drug_num_spatial', type=int, default=64)
    parser.add_argument('--drug_num_edge_dist', type=int, default=5)
    parser.add_argument('--drug_multi_hop_max_dist', type=int, default=5)
    parser.add_argument('--drug_n_encode_layers', type=int, default=12)
    parser.add_argument('--drug_embed_dim', type=int, default=80)
    parser.add_argument('--drug_num_heads', type=int, default=8)
    parser.add_argument('--layer_dropout_rate', type=float, default=0.)

    # ccl_enc
    parser.add_argument('--num_genes', type=int, default=4446)
    parser.add_argument('--in_features', type=int, default=1)
    parser.add_argument('--embed_dim_1', type=int, default=16)
    parser.add_argument('--num_pathways', type=int, default=186)
    parser.add_argument('--ccl_num_heads', type=int, default=8)
    parser.add_argument('--num_pattn_layers', type=int, default=4)
    
    parser.add_argument('--embed_dim', type=int, default=80)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--q_noise', type=float, default=0.)
    parser.add_argument('--qn_block_size', type=int, default=8)
    parser.add_argument('--cd_num_heads', type=int, default=8)
    parser.add_argument('--ccl_drug_trans', type=bool, default=True)
    
    # test
    parser.add_argument('--ft_steps', type=int, default=0)
    parser.add_argument('--ft_lr', type=float, default=1e-5)
    
    
    return parser.parse_args()

def main():
    
    args = arg_parse()
    set_random_seed(seed=_random_seed)
    args.device = "cuda"
    args.drug_data_dir = osp.join("../data_processed", args.drug_dataset_name)
    args.dr_data_dir = osp.join("../data_processed", args.drug_dataset_name, "%s_%s" % (args.drug_dataset_name, args.omics_combination))
    args.ccl_data_path = osp.join("../data_processed", args.ccl_dataset_name,"%s_%s_p%d.npy" % (args.ccl_dataset_name, args.omics_combination, args.num_pathways))
    
    args.task_name = "DRP_wPre_%s_p%d" % (args.omics_combination, args.num_pathways)
    if args.in_features != len(args.omics_combination):
        args.in_features = len(args.omics_combination)
    
    # train
    traditional_trainer = TraditionalBase(args)
    if args.mode == "train":
        dataloader = DRDataloderForTraditionalTraining(dataset_args=args)
        traditional_trainer.train(dataloader)
    
    # test
    elif args.mode == "test":
        
        if args.ft_steps > 0:
            dataloader = DRDataloderForFewShotLearning(args)
            metrics, logits_df = traditional_trainer.test_w_tf(dataloader)
            metrics_df = pd.DataFrame(metrics)
            logits_df.to_csv(osp.join("../model_logging", args.task_name, "%s_logits_k%d_wFT%d.csv" % (args.dataset_name_val, args.k, args.ft_steps)), index=None)
            metrics_df.to_csv(osp.join("../model_logging", args.task_name, "%s_metrics_k%d_wFT%d.csv" % (args.dataset_name_val, args.k, args.ft_steps)), index=None)
        else:
            dataloader = DRDataloderForTraditionalTraining(args)
            logits_df = traditional_trainer.test_wo_tf(dataloader)
            logits_df.to_csv(osp.join("../model_logging", args.task_name, "%s_logits_woFT.csv" % (args.dataset_name_val)), index=None)
            
            
if __name__ == "__main__":
    main()            
        
