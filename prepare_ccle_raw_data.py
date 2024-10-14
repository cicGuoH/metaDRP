# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:21:02 2024

@author: GUI
"""
import numpy as np
import pandas as pd
import os.path as osp

#%% 处理Exp数据

data_path = osp.join("../raw_data/CCL", "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
data = pd.read_csv(data_path, header=0, index_col=0)

gene_symbols = [i.split(" ")[0] for i in data.columns.values]
print(np.unique(gene_symbols).shape)

data.columns = gene_symbols
save_data_path = osp.join("../raw_data/CCL", "Exp.csv")
data.to_csv(save_data_path, index=None)


#%% 处理CNV数据
data_path = osp.join("../raw_data/CCL", "OmicsCNGene.csv")
data = pd.read_csv(data_path, header=0, index_col=0)
gene_symbols = [i.split(" ")[0] for i in data.columns.values]
print(np.unique(gene_symbols).shape)
data.columns = gene_symbols

save_data_path = osp.join("../raw_data/CCL", "CNV.csv")
data.to_csv(save_data_path, index=None)

#%% 处理CRISPR数据
data_path = "../raw_data/CCL/CRISPRGeneEffect.csv"
data = pd.read_csv(data_path, header=0, index_col=0)
gene_symbols = [i.split(" ")[0] for i in data.columns.values]
print(np.unique(gene_symbols).shape)
data.columns = gene_symbols
save_data_path = osp.join("../raw_data/CCL", "CRISPR.csv")
data.to_csv(save_data_path, index=None)

#%% 处理methy数据 
data_path = "../raw_data/CCL/CCLE_RRBS_TSS_1kb_20180614.txt"
data = pd.read_table(data_path, header=0, index_col=0, na_values=["     NA"])
print("num of genes:", data["gene"].unique().shape)
model_info = pd.read_csv("../raw_data/CCL/Model.csv", header=0, index_col=None)
data_sample_CCLENames = pd.DataFrame(data.columns.values[6:].tolist(), columns=["CCLEName"])
data_sample_CCLENames = pd.merge(data_sample_CCLENames, model_info.loc[:,["ModelID", "CCLEName"]], on="CCLEName", how="left")

data_tar = pd.concat([data["gene"],data.iloc[:, 6:]], axis=1)
data_grouped_mean = data_tar.groupby('gene').mean()
data_grouped_mean.columns = data_sample_CCLENames["ModelID"]

# 去除重复的ModelID
data_sample_CCLENames = data_sample_CCLENames.groupby("ModelID").filter(lambda x:len(x)==1)
data_grouped_mean = data_grouped_mean.loc[data_sample_CCLENames["ModelID"], :]
data_grouped_mean.index.name = None
data_grouped_mean = data_grouped_mean.T
data_grouped_mean.to_csv("../raw_data/CCL/Methy.csv", index=None)

#%% 处理Mutation数据
data_path = "../raw_data/CCL/OmicsSomaticMutations.csv"
data = pd.read_csv(data_path, header=0, index_col=None)
print(data.columns.values)

data["HugoSymbol"].unique().shape
# 过滤沉默突变
data_filter = data[data["VariantInfo"]!="SILENT"]
data_filter = data_filter.loc[:, ["ModelID", "HugoSymbol"]]
value_counts = data_filter.groupby(["ModelID", "HugoSymbol"]).value_counts()
value_counts = value_counts.reset_index()
value_counts.columns = ["ModelID", "HugoSymbol", "Value"]
mut_sum = value_counts.pivot_table(index='ModelID', columns='HugoSymbol', values='Value', fill_value=0)
mut_sum.to_csv("../raw_data/CCL/Mut.csv", index=True)

