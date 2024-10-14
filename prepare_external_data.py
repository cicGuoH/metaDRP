# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:14:58 2024

@author: GUI

1. 细胞系/患者：列为基因，行为患者
2. 药物响应数据：drug_ID, sample_ID, tissue_ID
"""

import numpy as np
import pandas as pd
from pubchempy import get_compounds
import time
import os.path as osp
import os
from prepare_drug_data import process_drugs
import time

_rnd_seed = 4132231

def get_drug_info(drug_name, ind=1):
    time.sleep(1)
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

#%% PDTX

dataset_name = "PDTX"
"""
exp_dat = pd.read_csv("..\\raw_data\\PDTX\\ExpressionSamples.txt", sep="\t", header=0, index_col=0).T
exp_sample_ids = list(exp_dat.index.values)
exp_model_ids = list(set([i.split("-")[0] for i in exp_sample_ids])) #AB521
# Exp：40个model，包含153个样本，与文献中记载的不同（39），多了一个AB521
exp_dat_model = pd.read_csv("..\\raw_data\\PDTX\\ExpressionModels.txt", sep="\t", header=0, index_col=0).T
exp_model_ids = list(exp_dat_model.index.values)

methy_dat = pd.read_csv("..\\raw_data\\PDTX\\PromoterMethylationSamples.txt", sep="\t", header=0, index_col=0).T
methy_sample_ids = list(methy_dat.index.values)
methy_model_ids = list(set([i.split("-")[0] for i in methy_sample_ids]))
# DNA methy：38个model，包含68个样本，与文献中记载的（33）不同，多了AB630,AB569,AB521,AB582,AB551
 

dr_dat = pd.read_csv("..\\raw_data\\PDTX\\DrugResponsesAUCSamples.txt", sep="\t", header=0, index_col=None)
dr_sample_ids = list(dr_dat["ID"].unique()) # 37个样本，20个model
dr_model_ids = list(set([i.split("-")[0] for i in dr_sample_ids]))
drug_name = list(dr_dat["Drug"].unique()) #包含104个药物

dr_dat_model = pd.read_csv("..\\raw_data\\PDTX\\DrugResponsesAUCModels.txt", sep="\t", header=0, index_col=None)
dr_model_ids = list(dr_dat_model["Model"].unique())
"""

exp_dat_model = pd.read_csv("..\\raw_data\\PDTX\\ExpressionModels.txt", sep="\t", header=0, index_col=0).T
exp_model_ids = list(exp_dat_model.index.values)
dr_dat_model = pd.read_csv("..\\raw_data\\PDTX\\DrugResponsesAUCModels.txt", sep="\t", header=0, index_col=None)
dr_model_ids = list(dr_dat_model["Model"].unique())
# 保留具有Exp的DR
dr_models_with_exp = list(set(dr_model_ids) & set(exp_model_ids))
dr_data_with_exp = dr_dat_model.loc[np.isin(dr_dat_model["Model"], dr_models_with_exp), :]

# 过滤样本数量少于15个的药物
dr_data_with_exp_filter = dr_data_with_exp.groupby("Drug").filter(lambda x:len(x)>=15)
dr_data_with_exp_filter.loc[:,"Tissue"] = "Breast"
dr_data_with_exp_filter["Model"].unique() #20
len(dr_data_with_exp_filter["Drug"].unique()) #82

pdtx_exp = exp_dat_model.loc[dr_models_with_exp, :]
pdtx_exp.to_csv("../raw_data/PDTX/Exp.csv")


# 获取对应DR的drug信息
drugs_with_exp = list(dr_data_with_exp_filter["Drug"].unique()) #82
result = pd.DataFrame(columns = ["drug_name","molecular_weight", 
                                 "molecular_formula", "isomeric_smile",
                                 "iupac_name", "cid"])
align_miss = []
for i, item in enumerate(drugs_with_exp):
    ind, result_item = get_drug_info(item)
    if ind == 1 :
        print(i, item)
        result = pd.concat([result, result_item], ignore_index=True)
    
    if ind == 0:
        align_miss.append(item)
        print("row %d is not hit" % i)
result["Drug_ID"] = ["drug_%d" % i for i in range(len(result))]        
result.to_csv("../raw_data/PDTX/drug_info_raw.csv", index=None)

result = pd.read_csv("../data_processed/PDTX/drug_info.csv", index_col=None)

DR_fi = pd.merge(dr_data_with_exp_filter, result.loc[:, ["drug_name", "Drug_ID"]], left_on="Drug",right_on="drug_name", how="inner")
DR_fi["LN_IC50"] = -1 * np.log10(DR_fi["iC50"])

DR_dat_fi = DR_fi.loc[:, ["Drug_ID", "Model", "LN_IC50"]]
DR_dat_fi.columns = ["Drug_ID", "Model_ID", "IC50"]

omics_combination = "E"
data_dir = "../data_processed/%s/%s_%s" % (dataset_name, dataset_name, omics_combination)
tissue_id = "tissue_7"
dr_info_list = []
for drug_id in DR_dat_fi["Drug_ID"].unique():
    target_dir = osp.join(data_dir, drug_id)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    dr_sub = DR_dat_fi.loc[DR_dat_fi["Drug_ID"]==drug_id, :]
    dr_info_list.append([drug_id, tissue_id, len(dr_sub)])
    dr_sub.to_csv(osp.join(target_dir, "%s-%s.csv"%(drug_id, tissue_id)), index=None)
dr_info = pd.DataFrame(dr_info_list, columns=["Drug_ID", "Tissue_ID", "Size"])
dr_info.to_csv(osp.join("../data_processed", dataset_name, "%s_%s"%(dataset_name, omics_combination), "PDTX.csv"), index=None)


#%% TCGA BRCA数据整理
tcga_patient_data_path = "..\\raw_data\\TCGA\\bioinformatics_32_19_2891_s1\\bioinfo16_supplementary_tables.xlsx"
tcga_patient_data = pd.read_excel(tcga_patient_data_path, sheet_name="Table S2", header=2, index_col=None)
tcga_patient_data = tcga_patient_data.iloc[1:, :]

print(tcga_patient_data.groupby("Cancer")["drug_name"].count())
# Breast invasive carcinoma (BRCA) 389
# 提取BRCA数据，保留药物-患者数量大于5的
brca_patient_drug_data = tcga_patient_data[tcga_patient_data["Cancer"]=="Breast invasive carcinoma (BRCA)"]
# brca_patient_drug_data = brca_patient_drug_data.groupby("drug_name").filter(lambda x:len(x)>5)
brca_patient_drug_ids = brca_patient_drug_data["bcr_patient_barcode"].unique()

# 提取患者的Exp数据
brca_exp_data = pd.read_table("..\\raw_data\\TCGA\\HiSeqV2", header=0, index_col=0)
# 保留癌症患者barcode结尾是01
brca_exp_data_filter = brca_exp_data.loc[:, [i.split("-")[-1]=="01" for i in brca_exp_data.columns]]
raw_ids = brca_exp_data_filter.columns.values.tolist()
brca_exp_data_filter.columns = [i[:-3] for i in raw_ids]
brca_exp_data_fi = brca_exp_data_filter.T
brca_exp_data_fi.to_csv("..\\raw_data\\TCGA\\Exp.csv")

drug17_test_data = pd.DataFrame(columns=["Drug_ID", "Sample_ID", "Response"])
drug17_test_data["Sample_ID"] = brca_exp_data_fi.index.values
drug17_test_data["Drug_ID"] = "drug_17"
drug17_test_data["Response"] = 0 # 占位
drug17_test_data.to_csv("..\\data_processed\\TCGA\\TCGA_E\\drug_17\\drug_17-tissue_7.csv", index=None)

drug45_test_data = pd.DataFrame(columns=["Drug_ID", "Sample_ID", "Response"])
drug45_test_data["Sample_ID"] = brca_exp_data_fi.index.values
drug45_test_data["Drug_ID"] = "drug_45"
drug45_test_data["Response"] = 0 # 占位
drug45_test_data.to_csv("..\\data_processed\\TCGA\\TCGA_E\\drug_45\\drug_45-tissue_7.csv", index=None)


# 构造药物信息
brca_patient_drug_data = brca_patient_drug_data.groupby("drug_name").filter(lambda x:len(x)>5)
drugs = np.unique(brca_patient_drug_data["drug_name"]).tolist()
result = pd.DataFrame(columns = ["drug_name","molecular_weight", 
                                 "molecular_formula", "isomeric_smile",
                                 "iupac_name", "cid"])
align_miss = []
for i, item in enumerate(drugs):
    ind, result_item = get_drug_info(item)
    
    if ind == 1 :
        print(i, item)
        result = pd.concat([result, result_item], ignore_index=True)
    
    if ind == 0:
        align_miss.append(item)
        print("row %d is not hit" % i)
    time.sleep(1)
result["Drug_ID"] = ["drug_%d" % i for i in range(len(result))]     

result.to_csv("..\\raw_data\\TCGA\\drug_info_raw.csv")
result = pd.read_csv("../data_processed/TCGA/drug_info.csv", index_col=None)

DR_fi = brca_patient_drug_data.loc[:, ["bcr_patient_barcode", "drug_name", "measure_of_response"]]
DR_fi.columns = ["Sample_ID", "drug_name", "measure_of_response"]
DR_fi = pd.merge(DR_fi, result.loc[:, ["drug_name", "Drug_ID"]], left_on="drug_name",right_on="drug_name", how="inner")
tmp = np.unique(DR_fi["measure_of_response"]).tolist()
DR_fi["Response"] = DR_fi["measure_of_response"].replace(tmp, [i for i in range(len(tmp))], inplace=False)
DR_dat_fi = DR_fi.loc[:, ["Drug_ID", "Sample_ID", "Response", "measure_of_response"]]

omics_combination = "E"
dataset_name = "TCGA"
data_dir = "../data_processed/%s/%s_%s" % (dataset_name, dataset_name, omics_combination)

tissue_id = "tissue_7"
dr_info_list = []
for drug_id in DR_dat_fi["Drug_ID"].unique():
    target_dir = osp.join(data_dir, drug_id)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    dr_sub = DR_dat_fi.loc[DR_dat_fi["Drug_ID"]==drug_id, :]
    dr_info_list.append([drug_id, tissue_id, len(dr_sub)])
    dr_sub.to_csv(osp.join(target_dir, "%s-%s.csv"%(drug_id, tissue_id)), index=None)
dr_info = pd.DataFrame(dr_info_list, columns=["Drug_ID", "Tissue_ID", "Size"])
dr_info.to_csv(osp.join("../data_processed", dataset_name, "%s_%s"%(dataset_name, omics_combination), "%s.csv"% dataset_name), index=None)

#%% TCGA pancancer 数据
tcga_dr_path = "..\\raw_data\\TCGA\\TCGA_DR.csv"
tcga_dr = pd.read_csv(tcga_dr_path, header=0, index_col=None)
tcga_dr["drug.name"].unique()

tcga_exp = pd.read_csv("..\\raw_data\\TCGA\\TCGA_EXP.csv", header=0, index_col=0)
tcga_sample_ids = tcga_exp.index.values

sampels_tar = list(set(tcga_dr["patient.arr"].unique()) & set(tcga_sample_ids))
# 保留具有exp的DR
tcga_dr_with_exp = tcga_dr.loc[np.isin(tcga_dr["patient.arr"], sampels_tar), :]
tcga_exp = tcga_exp.loc[sampels_tar, :]
tcga_exp.to_csv("..\\raw_data\\panTCGA\\Exp.csv")

# 获取对应DR的drug信息
drugs_with_exp = list(tcga_dr_with_exp["drug.name"].unique()) #82
result = pd.DataFrame(columns = ["drug_name","molecular_weight", 
                                 "molecular_formula", "isomeric_smile",
                                 "iupac_name", "cid"])
align_miss = []
for i, item in enumerate(drugs_with_exp):
    ind, result_item = get_drug_info(item)
    if ind == 1 :
        print(i, item)
        result = pd.concat([result, result_item], ignore_index=True)
    
    if ind == 0:
        align_miss.append(item)
        print("row %d is not hit" % i)
result["Drug_ID"] = ["drug_%d" % i for i in range(len(result))]        
result.to_csv("../raw_data/panTCGA/drug_info_raw.csv", index=None)

result = pd.read_csv("../data_processed/panTCGA/drug_info.csv", index_col=None)

DR_fi = pd.merge(tcga_dr_with_exp, result.loc[:, ["drug_name", "Drug_ID"]], left_on="drug.name",right_on="drug_name", how="inner")
DR_fi["Response"] = np.where(DR_fi["response_RS"]=="Response", 1, 0)

DR_dat_fi = DR_fi.loc[:, ["Drug_ID", "patient.arr", "Response","cancers"]]
DR_dat_fi.columns = ["Drug_ID", "Model_ID", "Response", "CancerType"]

tissue_mapping = pd.read_csv("../raw_data/CCL/CCL_info.csv", header=0, index_col=None).iloc[:, [-5,-1]].drop_duplicates()
DR_dat_fi = pd.merge(DR_dat_fi, tissue_mapping, left_on="CancerType", right_on="OncotreeCode", how="left")
DR_dat_fi = DR_dat_fi.dropna()

dataset_name = "panTCGA"
omics_combination = "E"
data_dir = "../data_processed/%s/%s_%s" % (dataset_name, dataset_name, omics_combination)
dr_info_list = []
for drug_id in DR_dat_fi["Drug_ID"].unique():
    target_dir = osp.join(data_dir, drug_id)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    dr_sub = DR_dat_fi.loc[DR_dat_fi["Drug_ID"]==drug_id, :]
    tissues_sub = dr_sub["Tissue_ID"].unique()
    for tissue in tissues_sub:
        dr_tissue_sub = dr_sub.loc[dr_sub["Tissue_ID"]==tissue]
        dr_info_list.append([drug_id, tissue, len(dr_tissue_sub)])
        dr_tissue_sub.to_csv(osp.join(target_dir, "%s-%s.csv"%(drug_id, tissue)), index=None)
dr_info = pd.DataFrame(dr_info_list, columns=["Drug_ID", "Tissue_ID", "Size"])
dr_info = pd.merge(dr_info, result[["Drug_ID", "src_Drug_ID"]], on="Drug_ID", how="left")
dr_info.to_csv(osp.join("../data_processed", dataset_name, "%s_%s"%(dataset_name, omics_combination), "panTCGA.csv"), index=None)

