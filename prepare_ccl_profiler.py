# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 13:32:17 2023

@author: GUI
"""

import numpy as np
import pandas as pd
import os, sys, time
import os.path as osp
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from data_utils.feature_propagation import filling
import networkx as nx
from scipy import sparse
from torch_geometric.transforms import ToDevice

class CCLProfiler:
    
    def __init__(self, 
                 omics_combination="E", 
                 raw_data_path = "../raw_data/CCL",
                 pathway_data_path = "../raw_data/MsigDB/c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt",
                 pathway_dir = "../data_processed/MSigDB_p186", 
                 to_be_scaled = True, 
                 padlen = 400, 
                 gene_limit = 10,
                 threshold = 0, 
                 ):
        
        self.omics_combination = omics_combination
        self.raw_data_path = raw_data_path
        self.pathway_data_path = pathway_data_path
        self.pathway_dir = pathway_dir
        self.to_be_scaled = to_be_scaled
        
        self.gene_subset, self.edge_index = self.get_gene_subset()
        self.data_dict = self.load_raw_featureMap(to_be_scaled)
        
    def process_data(self, filling_method="feature_propagation", num_iterations=40, device="cuda"):
        data_dict = self.load_raw_featureMap(self.to_be_scaled)
        ccl_names = data_dict["E"].index.values.tolist()
        ccl_names = set()
        for (i, omics) in enumerate(self.omics_combination):
            ccl_names = ccl_names.union(set(data_dict[omics].index.values))
            
        data_sub_dict = {}
        for (i, omics) in enumerate(self.omics_combination):    
           data_sub_dict[omics] = data_dict[omics].reindex(index=ccl_names, fill_value=np.nan).astype(np.float32)
        
        ccl_dict = {}
        for (ccl_ind, ccl_name) in enumerate(ccl_names):
            for i,omics in enumerate(self.omics_combination):
                if i == 0:
                    ccl_fea = torch.tensor(data_sub_dict[omics].loc[ccl_name, :], dtype=torch.float32).reshape((-1, 1))
                else:
                    next_fea = torch.tensor(data_sub_dict[omics].loc[ccl_name, :], dtype=torch.float32).reshape((-1, 1))
                    ccl_fea = torch.cat([ccl_fea, next_fea], dim=-1)
            mask = self.get_node_mask(ccl_fea)
            if filling_method is not None:
                start = time.time()
                ccl_fea = filling(filling_method=filling_method, edge_index=self.edge_index.to(device), X=ccl_fea.to(device), feature_mask=mask.to(device), num_iterations=num_iterations)
                print(f"ccl-{ccl_ind} Feature filling completed. It took: {time.time() - start:.2f}s")
            transform = ToDevice(device='cpu')
            ccl_dict[ccl_name] = transform(ccl_fea)
        return ccl_dict
    
    def get_gene_subset(self):
  
        pathway_mask_df = pd.read_csv(osp.join(self.pathway_dir, "pathway_mask.csv"), header=0, index_col=0)
        gene_subset = pathway_mask_df.columns.values.tolist()
        print("Module_Num: %d, Gene_Num: %d" % pathway_mask_df.shape)  
        edge_index = torch.load(osp.join(self.pathway_dir, "edge_index.pt"))
        if not osp.exists(osp.join(self.pathway_dir, "connectivity.pt")):
            pathway_gene_connectivity = self.get_pathway_gene_connectivity(pathway_mask_df)
            torch.save(pathway_gene_connectivity, osp.join(self.pathway_dir, "connectivity.pt"))
        return gene_subset, edge_index
    
    def get_pathway_gene_connectivity(self, pathway_mask_df):
        indices = np.array(np.nonzero(pathway_mask_df.values))
        indices = torch.tensor(indices)
        return indices
                
    def get_node_mask(self, feature_input):
       mask = ~torch.isnan(feature_input)
       return mask
    
    def load_raw_featureMap(self, to_be_scaled=True):
        data_dict = {}
        if "E" in self.omics_combination:
            exp_dat = pd.read_csv(osp.join(self.raw_data_path, "Exp.csv"), header=0, index_col=0)
            if to_be_scaled:
                scaler = StandardScaler()
                exp_dat = pd.DataFrame(scaler.fit_transform(exp_dat), index=exp_dat.index.values, columns=exp_dat.columns.values)
            exp_dat_sub = exp_dat.reindex(columns=self.gene_subset, index=exp_dat.index.values, fill_value=np.nan).astype(np.float32)
            data_dict["E"] = exp_dat_sub
        if "C" in self.omics_combination:
            cnv_dat = pd.read_csv(os.path.join(self.raw_data_path, "CNV.csv"), header=0, index_col=0)
            cnv_dat_sub = cnv_dat.reindex(columns=self.gene_subset, index=cnv_dat.index.values, fill_value=np.nan).astype(np.float32)
            data_dict["C"] = cnv_dat_sub
            
        if "D" in self.omics_combination:
            methy_dat = pd.read_csv(os.path.join(self.raw_data_path, "Methy.csv"), header=0, index_col=0)
            if to_be_scaled:
                scaler = StandardScaler()
                methy_dat = pd.DataFrame(scaler.fit_transform(methy_dat), index=methy_dat.index.values, columns=methy_dat.columns.values)
            methy_dat_sub = methy_dat.reindex(columns=self.gene_subset, index=methy_dat.index.values, fill_value=np.nan).astype(np.float32)
            data_dict["D"] = methy_dat_sub
            
        if "P" in self.omics_combination:
            crispr_dat = pd.read_csv(os.path.join(self.raw_data_path, "CRISPR.csv"), header=0, index_col=0)
            crispr_dat_sub = methy_dat.reindex(columns=self.gene_subset, index=crispr_dat.index.values, fill_value=np.nan).astype(np.float32)
            data_dict["P"] = crispr_dat_sub
        return data_dict

def get_pathway_ppi_geneset(pathway_data_path, threshold, gene_limit, ):
    pathway_dict = {}
    max_len = 0
    count = 0
    gene_subset = []
    with open(pathway_data_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            tmp_list = line.split("\t")
            if len(tmp_list[2:]) >= gene_limit:
                count += 1
                if len(tmp_list[2:]) > max_len:
                    max_len = len(tmp_list[2:])
                pathway_dict[tmp_list[0]] = tmp_list[2:]
                gene_subset.extend(tmp_list[2:])
    
    gene_subset = set(gene_subset)
    raw_ppi_df = pd.read_csv("../raw_data/PPI/PPI_with_symbols.csv", header=0, index_col=None)
    raw_ppi_graph = nx.from_pandas_edgelist(raw_ppi_df, source="partner1", target="partner2", edge_attr="combined_score")
    gene_withinPPI = set(raw_ppi_graph.nodes)
    gene_subset = list(set(gene_subset & gene_withinPPI))
    gene_subset.sort() # to make sure the same order
    sub_ppi_graph = nx.Graph(raw_ppi_graph.subgraph(gene_subset))
    
    # remove edges the weight of which smaller than threshold
    remove = [(i,j) for i,j,w in sub_ppi_graph.edges(data=True) if w["combined_score"] < threshold ]
    sub_ppi_graph.remove_edges_from(remove)
    sub_ppi_graph_df = nx.to_pandas_adjacency(sub_ppi_graph)
    sub_ppi_graph_df = sub_ppi_graph_df.loc[gene_subset, gene_subset]
    sparse_mx = sparse.csr_matrix(sub_ppi_graph_df.values).tocoo().astype(np.float32)
    edge_index = np.vstack((sparse_mx.row, sparse_mx.col))
    edge_index = torch.tensor(edge_index, dtype=torch.float32)
    pathway_mask_dict = {}
    for pathway in pathway_dict:
        pathway_mask_dict[pathway] = np.isin(gene_subset, pathway_dict[pathway])
        
    print("Module_Num: %d, Gene_Num: %d" % (count, len(gene_subset)))       
    pathway_mask_df = pd.DataFrame(pathway_mask_dict, index=gene_subset, dtype=np.int32).T
    pathway_mask_df.to_csv(osp.join("../data_processed/MSigDB_p%d"%count, "pathway_mask.csv"))
    torch.save(edge_index, osp.join("../data_processed/MSigDB_p%d"%count, "edge_index.pt"))    
    return count

if __name__ == "__main__":
    # omics_combination:["E", "EC", "ED", "ECD", "ECDP"]
    pathway_data_path = "../raw_data/MsigDB/c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt"
    num_pathways = get_pathway_ppi_geneset(pathway_data_path=pathway_data_path, threshold=0, gene_limit=10)
    dataset_name = "panTCGA"
    raw_data_path = "../raw_data/%s/" % dataset_name
    omics_combination = "E"
    save_path = "../data_processed/%s" % dataset_name
    filling_method = "feature_propagation"
    num_iterations = 40
    ccl_profiler = CCLProfiler(omics_combination=omics_combination, 
                                raw_data_path=raw_data_path,
                                pathway_data_path=pathway_data_path, 
                                pathway_dir="../data_processed/MSigDB_p%d"%num_pathways,
                                to_be_scaled=False)
    ccl_dict = ccl_profiler.process_data(filling_method=filling_method, num_iterations=num_iterations)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    np.save(osp.join(save_path, "%s_%s_p%d.npy" % (dataset_name, omics_combination, num_pathways)), ccl_dict)

