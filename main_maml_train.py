# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:57:14 2023

@author: GUI
"""

import argparse
import random
import os.path as osp
import numpy as np
from data_utils.dataset import DRDataloderForFewShotLearning    
import torch
from maml_train_system import MAMLBase

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
    
    parser.add_argument('--train_seed', type=int, default=0, help="random seed of sampling train dataset")
    parser.add_argument('--val_seed', type=int, default=0, help="random seed of sampling val/test dataset")
    parser.add_argument('--num_support_samples', type=int, default=10, help="number of support ccls per task")
    parser.add_argument('--num_query_samples', type=int, default=20, help="number of query ccls per task")
    parser.add_argument('--k', type=int, default=10, help="number of support ccls per task")
    parser.add_argument('--val_samples_per_batch', type=int, default=32, help="number of query ccls per task")
    parser.add_argument('--num_batches_per_iter', type=int, default=8, help="number of tasks per iter")
    parser.add_argument('--num_workers', type=int, default=5)
    
    # train
    parser.add_argument('--model_name', type=str, default="metaDRP", help="model name, etc. metaDRP")
    parser.add_argument('--meta_lr', type=float, default=1e-4)
    parser.add_argument('--inner_lr', type=float, default=1e-4)
    parser.add_argument('--num_inner_updates', type=int, default=4)
    parser.add_argument('--first_order', type=bool, default=True)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--total_epochs', type=int, default=1000)
    parser.add_argument('--pretrain_flag', action='store_false', help="whether to use drug pretrained model")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--patience', type=int, default=50)
    
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
    parser.add_argument('--num_pathways', type=int, default=186)
    parser.add_argument('--ccl_num_heads', type=int, default=8)
    parser.add_argument('--num_pattn_layers', type=int, default=1)
    
    parser.add_argument('--embed_dim', type=int, default=80)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--q_noise', type=float, default=0.)
    parser.add_argument('--qn_block_size', type=int, default=8)
    parser.add_argument('--cd_num_heads', type=int, default=16)
    parser.add_argument('--ccl_drug_trans', action='store_false')
    return parser.parse_args()



def main():
    args = arg_parse()
    set_random_seed(seed=_random_seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.drug_data_dir = osp.join("../data_processed", args.drug_dataset_name)
    args.dr_data_dir = osp.join("../data_processed", args.drug_dataset_name, "%s_%s" % (args.drug_dataset_name, args.omics_combination))
    args.ccl_data_path = osp.join("../data_processed", args.ccl_dataset_name,"%s_%s_p%d.npy" % (args.ccl_dataset_name, args.omics_combination, args.num_pathways))
    
    if args.pretrain_flag:
        args.task_name = "%s_wPre_%s_k%d_p%d_i%d" % (args.model_name, args.omics_combination, args.num_support_samples, args.num_pathways, args.num_inner_updates)
    else:
        args.task_name = "%s_%s_k%d_p%d_i%d" % (args.model_name, args.omics_combination, args.num_support_samples, args.num_pathways, args.num_inner_updates)
    
    args.mode = "train"
    if args.in_features != len(args.omics_combination):
        args.in_features = len(args.omics_combination)
    
    dr_dataloader = DRDataloderForFewShotLearning(args)
    
    gen_config = '\n'.join(['{}: {}'.format(k,v) for k,v in args.__dict__.items()])
    with open('../model_logging/%s/readme.txt' % args.task_name, 'w') as fw:
        fw.write(gen_config)
        
    # maml training
    maml_system = MAMLBase(args = args)
    maml_system.train(dataloader=dr_dataloader, valid=True)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
        
    
    
    
    
    