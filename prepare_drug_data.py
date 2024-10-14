# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:30:47 2023

@author: GUI
"""
import pandas as pd
import numpy as np
import os.path as osp
import os
from rdkit.Chem import AllChem as Chem
from tqdm import tqdm
from data_utils.drug_feature_ogb import smiles2graph, preprocess_item, preprocess_item_tsp
from sklearn.preprocessing import StandardScaler
from pubchempy import get_compounds
import time

_rnd_seed = 4132231

def get_drug_info(drug_name, ind=1):
    time.sleep(0.5)
    try:
        for compound in get_compounds(drug_name, 'name'):
            b1 = compound.cid
            c1 = compound.isomeric_smiles
            d1 = compound.molecular_formula
            e1 = compound.molecular_weight
            f1 = compound.iupac_name
        dataframe = pd.DataFrame({'drug_name':drug_name,
                                  'molecular_weight': e1,
                                  'molecular_formula': d1,
                                  'isomeric_smile': c1,
                                  'iupac_name': f1,
                                  'cid': b1}, index=[0])
        
    except:
        ind = 0
        dataframe = pd.DataFrame({'drug_name':drug_name,
                                  'molecular_weight': np.nan,
                                  'molecular_formula': np.nan,
                                  'isomeric_smile': np.nan,
                                  'iupac_name': np.nan,
                                  'cid': np.nan}, index=[0])
    
    return ind, dataframe


def align_drug_id(drug_list):
    result = pd.DataFrame(columns = ["drug_name", "molecular_weight", 
                                  "molecular_formula", "isomeric_smile",
                                  "iupac_name", "cid"])
    align_miss = []
    for i, tar in enumerate(drug_list):
        ind, result_item = get_drug_info(tar)
            
        if ind == 1 :
            print(i, tar)
            result = pd.concat([result, result_item], ignore_index=True)
        if ind == 0:
            align_miss.append(tar)
            print("row %d is not hit" % i)
    return result, align_miss


# GDSC1 药物数据库
def process_drugs(dataset_name="GDSC1", max_len=120):
    drug_dir = osp.join("../raw_data", dataset_name)
    drug_processed_dir = osp.join("../data_processed", dataset_name)
    
    if not osp.exists(drug_processed_dir):
        os.makedirs(drug_processed_dir)
        
    all_drug_info = pd.read_csv(osp.join(drug_dir, "drug_info_raw.csv"), header=0)
    # create drug_dict
    drug_dict = {}
    max_num_nodes = 0
    drug_id = 0
    all_drug_info["Drug_ID"] = np.nan
    for i in tqdm(range(len(all_drug_info))):
        drug_item = all_drug_info.iloc[i, :]
        drug_smile = drug_item["isomeric_smile"]
        canon_drug_smile = Chem.MolToSmiles(Chem.MolFromSmiles(drug_smile), isomericSmiles=True, canonical=True)
        
        if len(canon_drug_smile) <= max_len: 
            drug_graph = smiles2graph(canon_drug_smile)
            try:
                if drug_graph.x is not None:
                    drug_graph = preprocess_item(drug_graph)
                else:
                    drug_graph = preprocess_item_tsp(drug_graph)
                
                all_drug_info.iloc[i, -1] =  "drug_%d" % drug_id    
                drug_dict["drug_%d" % drug_id] = drug_graph
                drug_id += 1
                if drug_graph.num_nodes > max_num_nodes:
                    max_num_nodes = drug_graph.num_nodes
                    
            except:
                print("%d : %s fail to get drug graph" % (i, drug_smile))
    drug_info = all_drug_info.dropna(subset=["Drug_ID"])
    np.save(osp.join(drug_processed_dir, "drug_dict.npy"), drug_dict)
    drug_info.to_csv(osp.join(drug_processed_dir, "drug_info.csv"), index=None)         

def prepare_drug_tissue_data(dataset_name="GDSC1", omics_combination="E"):
    dr_data = pd.read_csv(osp.join("../raw_data", dataset_name, "drug_response.csv"), header=0, index_col=0)
    drug_info = pd.read_csv(osp.join("../data_processed", dataset_name, "drug_info.csv"), header=0, index_col=None).loc[:, ["GDSC1 drugs_IDs", "Drug_ID"]]
    drug_info.columns = ["Drug_Name", "Drug_ID"]
    ccl_info = pd.read_csv("../raw_data/CCL/CCL_info.csv", header=0).iloc[:, [0,-1]]
    ccl_dict = np.load("../data_processed/CCL/CCL_%s.npy" % omics_combination, allow_pickle=True).item()
    ccls_with_fea = list(ccl_dict.keys())
    ccl_info.columns = ["DepMap_ID", "Tissue_ID"]
    ccl_info = ccl_info.loc[np.isin(ccl_info["DepMap_ID"], ccls_with_fea), :]
    
    dr_info_list = []
    for i in tqdm(range(len(drug_info))):
        drug_name, drug_id = drug_info.iloc[i, :]
        dr_sub = dr_data.loc[:, drug_name].dropna()
        dr_sub = dr_sub.reset_index()
        dr_sub.columns = ["DepMap_ID", "IC50"]
        # scaler = StandardScaler()
        # dr_sub["IC50_scaled"] = scaler.fit_transform(dr_sub["IC50"].values.reshape(-1, 1)).reshape(-1)
        dr_sub["Drug_ID"] = drug_id
        dr_sub = pd.merge(dr_sub, ccl_info, how="inner", on="DepMap_ID")     
        tissues_sub = dr_sub["Tissue_ID"].unique()
        save_path = osp.join("../data_processed", dataset_name, "%s_%s"%(dataset_name, omics_combination), drug_id)
        if not osp.exists(save_path):
            os.makedirs(save_path)
        for tissue in tissues_sub:
            dr_tissue_sub = dr_sub.loc[dr_sub["Tissue_ID"]==tissue]
            # dr_tissue_sub = dr_tissue_sub.loc[:, ["Drug_ID", "DepMap_ID", "IC50", "IC50_scaled"]]
            dr_tissue_sub = dr_tissue_sub.loc[:, ["Drug_ID", "DepMap_ID", "IC50"]]
            dr_tissue_sub.to_csv(osp.join(save_path, "%s-%s.csv"%(drug_id, tissue)), index=None)
            dr_info_list.append([drug_id, tissue, len(dr_tissue_sub)])
    dr_info = pd.DataFrame(dr_info_list, columns=["Drug_ID", "Tissue_ID", "Size"])
    dr_info.to_csv(osp.join("../data_processed", dataset_name, "%s_%s"%(dataset_name, omics_combination), "drug_response_info.csv"), index=None)

def data_split(dataset_name="GDSC1", omics_combination="E", train_frac=0.85):
    rng = np.random.RandomState(_rnd_seed)
    drug_reponse_info = pd.read_csv(osp.join("../data_processed", dataset_name, "%s_%s"%(dataset_name, omics_combination), "drug_response_info.csv"), header=0)
    drugs = drug_reponse_info["Drug_ID"].unique()
    drugs_for_unseen = rng.choice(drugs, int(len(drugs)*0.1))
    
    # for meta-train and meta-val
    drug_response_info_sub = drug_reponse_info.loc[~np.isin(drug_reponse_info["Drug_ID"], drugs_for_unseen)]
    drug_response_info_sub1 = drug_response_info_sub.loc[drug_response_info_sub["Size"] >= 30]
    drug_response_info_sub2 = drug_response_info_sub.loc[drug_response_info_sub["Size"] < 30]
    
    sample_idx = [i for i in range(len(drug_response_info_sub1))]
    
    rng.shuffle(sample_idx)
    train_idx = sample_idx[:int(len(sample_idx)*train_frac)]
    test_idx = sample_idx[int(len(sample_idx)*train_frac):]
    dr_train = drug_response_info_sub1.iloc[train_idx, :]
    dr_test = drug_response_info_sub1.iloc[test_idx, :]
    dr_train.to_csv("../data_processed/%s/%s_%s/maml_train.csv" % (dataset_name, dataset_name, omics_combination), index=None)
    dr_test.to_csv("../data_processed/%s/%s_%s/maml_val.csv" % (dataset_name, dataset_name, omics_combination), index=None)
    # for few-shot validation
    unseen_data = drug_reponse_info.loc[np.isin(drug_reponse_info["Drug_ID"], drugs_for_unseen)]
    unseen_data.to_csv("../data_processed/%s/%s_%s/unseen_drug.csv" % (dataset_name, dataset_name, omics_combination), index=None)
    drug_response_info_sub2.to_csv("../data_processed/%s/%s_%s/unused_data.csv" % (dataset_name, dataset_name, omics_combination), index=None)

                    
if __name__ == "__main__":
    # drug ID align
    drug_info = pd.read_table("../raw_data/DRUG/Table_Drugs_Synonyms_cdb.txt", header=0)
    result = pd.DataFrame(columns = ["molecular_weight", 
                                  "molecular_formula", "isomeric_smile",
                                  "iupac_name", "cid"])
    align_miss = []
    for i, row in drug_info.iterrows():
        item = row["Drug_Synonyms"]
        tar = item.split(";")
        for j, tar_item in enumerate(tar):
            ind, result_item = get_drug_info(tar_item)
            
            if ind == 1 :
                print(i, tar_item)
                result = pd.concat([result, result_item], ignore_index=True)
                break
            
            if j == len(tar)-1:
                print(i, tar_item)
                result = pd.concat([result, result_item], ignore_index=True)
                if ind == 0:
                    align_miss.append(i)
                    print("row %d is not hit" % i)
    
    drug_info_addCID = pd.concat([drug_info, result], axis=1, ignore_index=False)
    drug_info_addCID.to_csv("../data_processed/DRUG/drug_info.csv", index=None)


    # drug_info = process_drugs(dataset_name="GDSC1")
    prepare_drug_tissue_data(dataset_name="GDSC1", omics_combination="EC")
    data_split(dataset_name="GDSC1", omics_combination="EC", train_frac=0.85)
    
    # generate PDTX drug_dict.npy
    drug_info = process_drugs(dataset_name="panTCGA")
