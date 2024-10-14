# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import os.path as osp
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch

from data_utils.drug_feature_ogb import smiles2graph, preprocess_item, preprocess_item_tsp
from data_utils.collator import collator_forDrug, collate_fn_meta, collate_fn_zero_shot, collator_fn_traditional
from torch_geometric.data import InMemoryDataset


# ZINC 250K
class PyGZINCDataset(InMemoryDataset):
    def __init__(self, root="../DataProcessed/ZINC", subset=True, split="train", smiles2graph=smiles2graph, transform=None, pre_transform = None):
        self.root = root
        self.subset = subset
        self.smiles2graph = smiles2graph
        super().__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    @property
    def raw_file_names(self):
        return '250k_rndm_zinc_drugs_clean_3.csv'
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'subset', 'processed')
    
    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names))
        smiles_list = data_df['smiles']
        logp_list = data_df['logP']
        print('Converting SMILES strings into graphs...')
        
        if not self.subset:
            train_data_list = []
            test_data_list = []
            test_idx = self.get_split_idx()
            
            for i in tqdm(range(len(smiles_list))):
                
                mol_graph = self.smiles2graph(smiles_list[i])
                mol_graph.y = torch.tensor([logp_list[i]])
                if i in test_idx:
                    test_data_list.append(mol_graph)
                else:
                    train_data_list.append(mol_graph)
            train_data, train_slices = self.collate(train_data_list)
            test_data, test_slices = self.collate(test_data_list)
            
            print('Saving...')
            torch.save((train_data, train_slices), self.processed_paths[0])
            torch.save((test_data, test_slices), self.processed_paths[2])
            
        else:
            train_data_list = []
            val_data_list = []
            test_data_list = []
            
            train_index = self.get_split_idx("train")
            val_index = self.get_split_idx("val")
            test_index = self.get_split_idx("test")
            
            
            for i in tqdm(range(len(train_index))):
                
                mol_graph = self.smiles2graph(smiles_list[train_index[i]])
                mol_graph.y = torch.tensor([logp_list[train_index[i]]])
                train_data_list.append(mol_graph)
            
            for i in tqdm(range(len(val_index))):
                
                mol_graph = self.smiles2graph(smiles_list[val_index[i]])
                mol_graph.y = torch.tensor([logp_list[val_index[i]]]) 
                val_data_list.append(mol_graph)
            
            for i in tqdm(range(len(test_index))):
                
                mol_graph = self.smiles2graph(smiles_list[test_index[i]])
                mol_graph.y = torch.tensor([logp_list[test_index[i]]])
                test_data_list.append(mol_graph)
            
            train_data, train_slices = self.collate(train_data_list)
            val_data, val_slices = self.collate(val_data_list)
            test_data, test_slices = self.collate(test_data_list)
            
            print('Saving...')
            torch.save((train_data, train_slices), self.processed_paths[0])
            torch.save((val_data, val_slices), self.processed_paths[1])
            torch.save((test_data, test_slices), self.processed_paths[2])

    def get_split_idx(self, split=None):
        if not self.subset:
            indices = np.load(osp.join(self.raw_dir, "test_idx.npy"))
        else:
            with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                indices = [int(x) for x in f.read()[:-1].split(',')]
        return indices
    

class BatchedDrugDataset(Dataset):
    def __init__(self, dataset, max_nodes=64, multi_hop_max_dist=5, spatial_pos_max=64):
        super().__init__()
        self.dataset = dataset
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        if item.x is not None:
            item = preprocess_item(item)
        else:
            item = preprocess_item_tsp(item)
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collator_forDrug(
            samples,
            max_node=self.max_nodes,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


class DatasetForMetaTrain(Dataset):
    def __init__(self, dataset_args, dataset_name="maml_train"):
        self.drug_data_dir = dataset_args.drug_data_dir
        self.dr_data_dir = dataset_args.dr_data_dir
        self.ccl_data_path = dataset_args.ccl_data_path
        self.drug_tissue_info = pd.read_csv(osp.join(self.dr_data_dir, "%s.csv" % dataset_name))
        
        self.init_seed = dataset_args.train_seed
        self.seed = dataset_args.train_seed
        self.num_support_samples = dataset_args.num_support_samples
        self.num_query_samples = dataset_args.num_query_samples
        
        self.ccl_dict, self.drug_dict = self.load_featuremap()
        self.data_length = len(self.drug_tissue_info)
        
        self.sample_name = dataset_args.sample_name
        self.measurement_method = dataset_args.measurement_method
        
    def load_featuremap(self):
        drug_dict = np.load(osp.join(self.drug_data_dir, "drug_dict.npy"), allow_pickle=True).item()
        ccl_dict = np.load(osp.join(self.ccl_data_path), allow_pickle=True).item()
        drug_target = self.drug_tissue_info["Drug_ID"].unique()
        drug_dict = {key: drug_dict[key] for key in drug_target if key in drug_dict}
        return ccl_dict, drug_dict
    
    def __len__(self):
        return self.data_length
    
    def get_set(self, drug_subset, rnd_seed):
        
        rng = np.random.RandomState(rnd_seed)
        sample_idx = rng.choice([i for i in range(len(drug_subset))], self.num_support_samples+self.num_query_samples)
        drug_id = drug_subset["Drug_ID"][0]
        support_df = drug_subset.iloc[sample_idx[:self.num_support_samples], :]
        query_df = drug_subset.iloc[sample_idx[self.num_support_samples:], :]
        
        drug_fea = self.drug_dict[drug_id]
        support_ccl_fea = [self.ccl_dict[ccl_item] for ccl_item in support_df[self.sample_name]]
        query_ccl_fea = [self.ccl_dict[ccl_item] for ccl_item in query_df[self.sample_name]]
        
        support_y = torch.tensor(support_df[self.measurement_method].values.tolist())
        query_y = torch.tensor(query_df[self.measurement_method].values.tolist())
        support_ccl_fea = torch.stack(support_ccl_fea, dim=0)
        query_ccl_fea = torch.stack(query_ccl_fea, dim=0)
        
        support_drug_fea = collator_forDrug([drug_fea for i in range(self.num_support_samples)])
        query_drug_fea = collator_forDrug([drug_fea for i in range(self.num_query_samples)])
        return support_drug_fea, support_ccl_fea, support_y, query_drug_fea, query_ccl_fea, query_y
    
    def __getitem__(self, index):
        self.seed = self.seed + index
        drug_item, tissue_item, _ = self.drug_tissue_info.iloc[index, :]
        drug_subset = pd.read_csv(osp.join(self.dr_data_dir, drug_item, "%s-%s.csv"%(drug_item, tissue_item)))
        support_drug_fea, support_ccl_fea, support_y, query_drug_fea, \
            query_ccl_fea, query_y = self.get_set(drug_subset=drug_subset, rnd_seed=self.seed)
        return support_drug_fea, support_ccl_fea, support_y, query_drug_fea, query_ccl_fea, query_y
    
    def update_seed(self, current_iter):
        self.seed = self.init_seed +  current_iter
    
    def reset_seed(self):
        self.seed = self.init_seed
    
    
class DatasetForFewShot(Dataset):
    def __init__(self, dataset_args, dataset_name="maml_test"):
        self.drug_data_dir = dataset_args.drug_data_dir
        self.dr_data_dir = dataset_args.dr_data_dir
        self.ccl_data_path = dataset_args.ccl_data_path
        
        self.sample_name = dataset_args.sample_name
        self.measurement_method = dataset_args.measurement_method
        
        self.seed = dataset_args.val_seed #控制选择哪些support data
        self.k = dataset_args.k
        self.val_samples_per_batch = dataset_args.val_samples_per_batch
        
        drug_tissue_info = pd.read_csv(osp.join(self.dr_data_dir, "%s.csv" % dataset_name))
        self.drug_tissue_info = drug_tissue_info[drug_tissue_info["Size"] >= self.k + 1]
        print("Num of drug-tissue tasks is %d/%d" % (len(drug_tissue_info), len(self.drug_tissue_info)))
        self.data_length = len(self.drug_tissue_info)
        self.ccl_dict, self.drug_dict = self.load_featuremap()
        
    def load_featuremap(self):
        drug_dict = np.load(osp.join(self.drug_data_dir, "drug_dict.npy"), allow_pickle=True).item()
        ccl_dict = np.load(osp.join(self.ccl_data_path), allow_pickle=True).item()
        drug_target = self.drug_tissue_info["Drug_ID"].unique()
        drug_dict = {key: drug_dict[key] for key in drug_target if key in drug_dict}
        return ccl_dict, drug_dict
    
    def get_set_minibatch(self, drug_subset, rnd_seed, ccl_graph=True):
        """
        相比与原版的修改是，全部更改为使用minibatch，毕竟一个batch也是batch，不需要额外对很小的测试集合进行处理
        """
        rng = np.random.RandomState(rnd_seed)
        drug_id = drug_subset["Drug_ID"][0]
        drug_fea = self.drug_dict[drug_id]
        
        sample_idx = [i for i in range(len(drug_subset))]
        rng.shuffle(sample_idx)
        support_df =  drug_subset.iloc[sample_idx[:self.k], :]
        support_ccl_fea = [self.ccl_dict[ccl_item] for ccl_item in support_df[self.sample_name]]
        support_ccl_fea = torch.stack(support_ccl_fea, dim=0)
        support_drug_fea = collator_forDrug([drug_fea for i in range(self.k)]) 
        support_y = torch.tensor(support_df[self.measurement_method].values.tolist())
        
        tmp_num_mini_batch = len(sample_idx[self.k:]) // self.val_samples_per_batch
        num_mini_batch =  tmp_num_mini_batch if len(sample_idx[self.k:]) % self.val_samples_per_batch == 0 else (tmp_num_mini_batch + 1)
        
           
        mini_batch_query_ccl_fea = []
        mini_batch_query_drug_fea = []
        mini_batch_query_y = []
        for i in range(num_mini_batch):
            start_ind = self.k + i * self.val_samples_per_batch
            end_ind = min(self.k + (i+1) * self.val_samples_per_batch, len(sample_idx))
            if (len(sample_idx) - end_ind) <= 4:
                end_ind = len(sample_idx)
            query_df = drug_subset.iloc[sample_idx[start_ind:end_ind], :]
            query_ccl_fea = [self.ccl_dict[ccl_item] for ccl_item in query_df[self.sample_name]]
            query_ccl_fea = torch.stack(query_ccl_fea, dim=0)
            query_y = torch.tensor(query_df[self.measurement_method].values.tolist())
            query_drug_fea = collator_forDrug([drug_fea for i in range(len(query_df))])
            mini_batch_query_ccl_fea.append(query_ccl_fea)
            mini_batch_query_drug_fea.append(query_drug_fea)
            mini_batch_query_y.append(query_y)
            if end_ind == len(sample_idx):
                break
        return support_drug_fea, support_ccl_fea, support_y, mini_batch_query_drug_fea, mini_batch_query_ccl_fea, mini_batch_query_y
    
    def __getitem__(self, index):
        drug_item, tissue_item, _ = self.drug_tissue_info.iloc[index, :]
        drug_subset = pd.read_csv(osp.join(self.dr_data_dir, drug_item, "%s-%s.csv"%(drug_item, tissue_item)))
        support_drug_fea, support_ccl_fea, support_y, mini_batch_query_drug_fea, mini_batch_query_ccl_fea, \
            mini_batch_query_y = self.get_set_minibatch(drug_subset=drug_subset, rnd_seed=self.seed)
        return support_drug_fea, support_ccl_fea, support_y, mini_batch_query_drug_fea, mini_batch_query_ccl_fea, mini_batch_query_y
   
        
    def __len__(self):
        return self.data_length
    
    def get_sample_df(self, drug_index):
        drug_item, tissue_item, _ = self.drug_tissue_info.iloc[drug_index, :]
        drug_subset = pd.read_csv(osp.join(self.dr_data_dir, drug_item, "%s-%s.csv"%(drug_item, tissue_item)))
        sample_idx = [i for i in range(len(drug_subset))]
        rng = np.random.RandomState(self.seed)
        rng.shuffle(sample_idx)
        support_df = drug_subset.iloc[sample_idx[:self.k], :]
        query_df = drug_subset.iloc[sample_idx[self.k:], :]
        return drug_item, tissue_item, support_df, query_df
    

class DatasetForZeroShot(Dataset):
    def __init__(self, dataset_args, dataset_name="maml_test"):
        self.drug_data_dir = dataset_args.drug_data_dir
        self.dr_data_dir = dataset_args.dr_data_dir
        self.ccl_data_path = dataset_args.ccl_data_path
        self.seed = dataset_args.val_seed #数据集打乱
        self.val_samples_per_batch = dataset_args.val_samples_per_batch
        
        self.sample_name = dataset_args.sample_name
        self.measurement_method = dataset_args.measurement_method
        
        self.drug_tissue_info = pd.read_csv(osp.join(self.dr_data_dir, "%s.csv" % dataset_name))
        self.data_length = len(self.drug_tissue_info)
        self.ccl_dict, self.drug_dict = self.load_featuremap()
        
        
    def load_featuremap(self):
        drug_dict = np.load(osp.join(self.drug_data_dir, "drug_dict.npy"), allow_pickle=True).item()
        ccl_dict = np.load(osp.join(self.ccl_data_path), allow_pickle=True).item()
        drug_target = self.drug_tissue_info["Drug_ID"].unique()
        drug_dict = {key: drug_dict[key] for key in drug_target if key in drug_dict}
        return ccl_dict, drug_dict
    
    def get_set_minibatch(self, drug_subset, rnd_seed, ccl_graph=True):
        rng = np.random.RandomState(rnd_seed)
        drug_id = drug_subset["Drug_ID"][0]
        drug_fea = self.drug_dict[drug_id]
        
        sample_idx = [i for i in range(len(drug_subset))]
        rng.shuffle(sample_idx)
        
        tmp_num_mini_batch = len(sample_idx) // self.val_samples_per_batch
        num_mini_batch =  tmp_num_mini_batch if len(sample_idx) % self.val_samples_per_batch == 0 else (tmp_num_mini_batch + 1)
        
           
        mini_batch_query_ccl_fea = []
        mini_batch_query_drug_fea = []
        mini_batch_query_y = []
        for i in range(num_mini_batch):
            start_ind = i * self.val_samples_per_batch
            end_ind = min((i+1) * self.val_samples_per_batch, len(sample_idx))
            if (len(sample_idx) - end_ind) <= 4:
                end_ind = len(sample_idx)
            query_df = drug_subset.iloc[sample_idx[start_ind:end_ind], :]
            query_ccl_fea = [self.ccl_dict[ccl_item] for ccl_item in query_df[self.sample_name]]
            query_ccl_fea = torch.stack(query_ccl_fea, dim=0)
            query_y = torch.tensor(query_df[self.measurement_method].values.tolist())
            query_drug_fea = collator_forDrug([drug_fea for i in range(len(query_df))])
            mini_batch_query_ccl_fea.append(query_ccl_fea)
            mini_batch_query_drug_fea.append(query_drug_fea)
            mini_batch_query_y.append(query_y)
            if end_ind == len(sample_idx):
                break
        return mini_batch_query_drug_fea, mini_batch_query_ccl_fea, mini_batch_query_y
    
    
    def __getitem__(self, index):
        drug_item, tissue_item, _ = self.drug_tissue_info.iloc[index, :]
        drug_subset = pd.read_csv(osp.join(self.dr_data_dir, drug_item, "%s-%s.csv"%(drug_item, tissue_item)))
        mini_batch_query_drug_fea, mini_batch_query_ccl_fea, mini_batch_query_y = self.get_set_minibatch(drug_subset=drug_subset, rnd_seed=self.seed)
        return mini_batch_query_drug_fea, mini_batch_query_ccl_fea, mini_batch_query_y


    def __len__(self):
        return self.data_length
    
    def get_sample_df(self, drug_index):
        drug_item, tissue_item, _ = self.drug_tissue_info.iloc[drug_index, :]
        drug_subset = pd.read_csv(osp.join(self.dr_data_dir, drug_item, "%s-%s.csv"%(drug_item, tissue_item)))
        sample_idx = [i for i in range(len(drug_subset))]
        rng = np.random.RandomState(self.seed)
        rng.shuffle(sample_idx)
        query_df = drug_subset.iloc[sample_idx, :]
        return drug_item, tissue_item, query_df

    
class DRDataloderForFewShotLearning:
    def __init__(self, dataset_args):
        self.mode = dataset_args.mode # "train", "test"
        self.num_workers = dataset_args.num_workers
        
        if self.mode == "train":
            self.dataset_name_train = dataset_args.dataset_name_train
            self.dataset_name_val = dataset_args.dataset_name_val
            self.dataset_train = DatasetForMetaTrain(dataset_args, self.dataset_name_train)
            self.dataset_val = DatasetForFewShot(dataset_args, self.dataset_name_val)
            self.num_batches_per_iter = dataset_args.num_batches_per_iter
            self.total_epochs = dataset_args.total_epochs
            self.resume_epoch = dataset_args.resume_epoch
            self.total_train_iters_produced = 0
            self.num_iters_per_epoch = self.dataset_train.data_length//self.num_batches_per_iter
        
        elif self.mode == "test":
            self.dataset_name_val = dataset_args.dataset_name_val
            self.dataset_val = DatasetForFewShot(dataset_args, self.dataset_name_val)
            self.num_iters_per_epoch = self.dataset_val.data_length
            
    def get_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.num_batches_per_iter, collate_fn=collate_fn_meta, 
                      shuffle=True, num_workers=self.num_workers, drop_last=True) # shuffle设置为True，保证任务的随机
        return dataloader
    
    def get_dataloader_test(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_meta, 
                      shuffle=False, num_workers=self.num_workers, drop_last=False)
        return dataloader
    
    def get_train_batches(self):
        self.total_train_iters_produced += self.resume_epoch * self.num_iters_per_epoch
        self.dataset_train.update_seed(self.total_train_iters_produced-1)
        self.total_train_iters_produced += self.num_iters_per_epoch
        for sample_id, sample_batched in enumerate(self.get_dataloader(self.dataset_train)):
            yield sample_batched
    
    def get_val_batches(self):
        self.dataloader_length = self.dataset_val.data_length
        for sample_id, sample_batched in enumerate(self.get_dataloader_test(self.dataset_val)):
            yield sample_batched


class DRDataloderForZeroShotLearning:
    def __init__(self, dataset_args):
        self.dataset_name_val = dataset_args.dataset_name_val
        self.num_workers = dataset_args.num_workers
        
        self.dataset_val = DatasetForZeroShot(dataset_args, self.dataset_name_val)
        self.num_iters_per_epoch = self.dataset_val.data_length
            
    
    def get_dataloader_test(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_zero_shot, 
                      shuffle=False, num_workers=self.num_workers, drop_last=False)
        return dataloader
    
    def get_val_batches(self):
        self.dataloader_length = self.dataset_val.data_length
        for sample_id, sample_batched in enumerate(self.get_dataloader_test(self.dataset_val)):
            yield sample_batched    
            
            
class DatasetForTraditionalTrain(Dataset):
    def __init__(self, dataset_args, dataset_name="maml_train"):
        
        self.drug_data_dir = dataset_args.drug_data_dir
        self.dr_data_dir = dataset_args.dr_data_dir
        self.dr_data = self.get_dr_data(dataset_name)
        self.ccl_data_path = dataset_args.ccl_data_path
        self.ccl_dict, self.drug_dict = self.load_featuremap()
        self.seed = dataset_args.rm_seed
        self.init_seed = dataset_args.rm_seed
        self.data_length = len(self.dr_data)
        
    def get_dr_data(self, dataset_name):
        drug_tissue_data = pd.read_csv(osp.join(self.dr_data_dir, "%s.csv" % dataset_name))
        
        for i, item in drug_tissue_data.iterrows():
            drug_item = item["Drug_ID"]
            tissue_item = item["Tissue_ID"]
            
            if i==0:
                dr_data = pd.read_csv(osp.join(self.dr_data_dir, drug_item, "%s-%s.csv"%(drug_item, tissue_item)))
            else:
                tmp_data = pd.read_csv(osp.join(self.dr_data_dir, drug_item, "%s-%s.csv"%(drug_item, tissue_item)))
                dr_data = pd.concat([dr_data, tmp_data], axis=0, ignore_index=True)
        return dr_data
    
    def load_featuremap(self):
        drug_dict = np.load(osp.join(self.drug_data_dir, "drug_dict.npy"), allow_pickle=True).item()
        ccl_dict = np.load(osp.join(self.ccl_data_path), allow_pickle=True).item()
        return ccl_dict, drug_dict
    
    def __getitem__(self, index):
        drug_item, ccl_item, label = self.dr_data.iloc[index, :3]
        drug_fea = self.drug_dict[drug_item]
        ccl_fea = self.ccl_dict[ccl_item]
        y = torch.tensor(label).view((1,))
        return drug_fea, ccl_fea, y
        
    def __len__(self):
        return self.data_length
    
    # shuffle dataset every epoch
    def update_seed(self, current_epoch):
        self.seed = self.init_seed + current_epoch
        self.dr_data.sample(frac=1.0, replace=True, random_state=self.seed, ignore_index=True)
        
        
class DRDataloderForTraditionalTraining:
    def __init__(self, dataset_args):
        self.dataset_name_train = dataset_args.dataset_name_train
        self.dataset_name_val = dataset_args.dataset_name_val
        self.total_epochs = dataset_args.total_epochs
        self.resume_epoch = dataset_args.resume_epoch
        self.num_workers = dataset_args.num_workers
        self.num_batches_per_iter = dataset_args.num_batches_per_iter
        
        self.dataset_train = DatasetForTraditionalTrain(dataset_args, self.dataset_name_train)
        self.dataset_val = DatasetForTraditionalTrain(dataset_args, self.dataset_name_val)
        self.total_train_iters_produced = 0
        self.num_iters_per_epoch = self.dataset_train.data_length//self.num_batches_per_iter
        
        
    def get_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.num_batches_per_iter, collate_fn=collator_fn_traditional, 
                      shuffle=False, num_workers=self.num_workers, drop_last=True) # manually generate data batches for reproduction
        return dataloader
    
    
    def get_dataloader_val(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.num_batches_per_iter, collate_fn=collator_fn_traditional, 
                      shuffle=False, num_workers=self.num_workers, drop_last=False) # manually generate data batches for reproduction
        return dataloader


    def get_train_batches(self):
        self.total_train_iters_produced += self.resume_epoch * self.num_iters_per_epoch
        self.current_epoch = self.resume_epoch
        self.dataset_train.update_seed(self.current_epoch)
        self.total_train_iters_produced += self.num_iters_per_epoch
        self.current_epoch += 1
        for sample_id, sample_batched in enumerate(self.get_dataloader(self.dataset_train)):
            yield sample_batched

    
    def get_val_batches(self):
        for sample_id, sample_batched in enumerate(self.get_dataloader_val(self.dataset_val)):
            yield sample_batched
            