# metaDRP
In this work, we developed metaDRP, an interpretable deep learning framework metaDRP designed to predict drug responses using a few-shot learning approach. Our findings demonstrate the robustness of metaDRP across diverse datasets and predictive tasks, particularly in environments with limited data. By leveraging the MAML framework, metaDRP exhibited superior performance over traditional machine learning methods, particularly in few-shot learning scenarios. The model's ability to generalize from minimal data points and its effectiveness in distinguishing drug responses underscore its potential in personalized medicine applications.

## model overview

![Figure S1](https://github.com/user-attachments/assets/8accfdcf-cc32-4835-98d7-2b7c2076886d)
- Cell line branch
  - For a given cell line or tumor sample $G^c=(V^c,E^c,X^c )$, $V^c$ denotes 4446 genes, $E^c$ represents the relationships between genes, and $X^c$ represents the features of the nodes. We employed three layers of graph convolutional networks (GCN) to extract and aggregate gene-level information from the molecular profiles of the cell line or tumor. This configuration enabled nodes to aggregate information from their 3-hop neighbors, thereby expanding the receptive field of GCN. Residual connections were implemented at each layer to mitigate the issue of over-smoothing. After extracting gene-level information, we utilized a sparse linear layer in lieu of typical graph neural aggregation layers to consolidate gene-level information into pathway-level representations. Each neuron in this layer is sparsely connected to a corresponding KEGG pathway, enhancing the model's specificity. To further boost the model's associative learning capabilities, self-attention mechanisms were employed to integrate pathway-level features, ultimately yielding pathway-level representations $z_c\in R^{186×n_e}$. The parameters of the cell line branch encoder were configured as $W_c$。
- Drug branch
  - the SLIM version of the Graphormer architecture pre-trained on the ZINC 250k dataset. Given a molecular graph $G^d$，we derived the drug embeddings $z_d∈R^{n_e}$ from the drug branch with the parameters set as $W_d$.
- Predictor
  - The predictor was composed of two components: the fusion of multi-modal information between the cell line/tumor and the drug, and the feedforward prediction module, as illustrated in Figure S1. After representation learning was completed for the cell line/tumor and the drug in their respective branches, the multi-modal information fusion was achieved by a cross-attention mechanism and a feedforward network $FNN_1$ resulting in a joint representation $z_{cd} \in R^{n_e}$. Subsequently, the drug response is predicted using a second feedforward network $FNN_2$. The parameters for the predictor were established as $W_p$.
## Create the conda environment
```shell
conda env create -f environment.yml
```

## How to use

### Train a meta-learner with different configures. 
Configures available:
  - `pretrain_flag`: action="store_false". if not specified, the meta-learner is compiled with a pretrained graphormer.
  - `omics_combination`: E, EC, ECD, ECDP (E: gene_expression, C: CNVB, D: DNA methylation, P: CRISPR gene effect score)
  - `resume_epoch`: retrain the model from the specified epoch, defalut 0.
  - `sample_name`: to specify the title of samples, e.g. the title of CCL samples is DepMap_ID, while that of TCGA samples is Sample_ID
  - `num_inner_updates`: to specify the number of inner updates
  - `measurement_method`: to specify the measurement method of different cohorts, e.g. IC50 for drug screenin dataset, Response for TCGA patients.
    
```shell
# default, used to train a meta-learner.
nohup python -u main_maml_train.py --resume_epoch 0 --k 10 --num_inner_updates 2 --num_support_samples 10 --num_query_samples 20 --omics_combination E --sample_name DepMap_ID --measurement_method IC50 --ccl_dataset_name CCL > ../tmp_logging/0720_train_k10_E_i2.out 2>&1 &

# without pretrain
nohup python -u main_maml_train.py --resume_epoch 0 --k 10 --num_inner_updates 3 --pretrain_flag --num_support_samples 10 --num_query_samples 20 --omics_combination E --sample_name DepMap_ID --measurement_method IC50 --ccl_dataset_name CCL > ../tmp_logging/train_woPre_k10_E_i4.out 2>&1 &
```

### Predict samples using well-trained meta-learner
- using k to specify the number of support samples for k-shot fine-tuning.
    
```shell
nohup python -u main_maml_test.py --dataset_name_val unseen_drug --k 1 --num_inner_updates 2 --omics_combination E --sample_name DepMap_ID --val_seed 1 --measurement_method IC50 --ccl_dataset_name CCL --drug_dataset_name GDSC1 > ../tmp_logging/unseen_drug_k1_i2.out 2>&1 &

# predict TCGA patients with a well-trained meta-learner without fine tuning.
nohup python -u main_maml_test.py --dataset_name_val TCGA --k 0 --num_inner_updates 2 --omics_combination E --sample_name Sample_ID --measurement_method Response --ccl_dataset_name TCGA --drug_dataset_name TCGA --return_metrics > ../tmp_logging/test_TCGA_k0_i2.out 2>&1 &
```

### Using model explainer to extract the attention scores or embeddings of given drug-sample pairs.
- Using 10 samples regard to drug_265 and tissue_5 to fune-tune the meta-learner. Specify the repeating number `num_repeats`
```shell
python -u main_model_explain.py --drug_id 265 --tissue_id 5 --num_repeats 10 --k 10 --num_support_samples 10 --num_inner_updates 2
```
- Using CCL samples to fine-tune the model and then predict or explain the predictions of TCGA patients. In case the Drug_ID is not aligned, specify the `drug_id` for the target samples and `src_drug_id` for the support samples.
```shell
python -u main_model_croxexplain.py --drug_id 45 --tissue_id 7 --num_repeats 10 --k 10 --drug_dataset_name TCGA --ccl_dataset_name TCGA --omics_combination E --sample_name Sample_ID

python -u main_model_croxexplain.py --drug_id 1 --tissue_id 7 --src_drug_id 14 --num_repeats 10 --k 10 --drug_dataset_name panTCGA --ccl_dataset_name panTCGA --omics_combination E --sample_name Sample_ID
```
- Help visualize the embeddings and attention scores.
```shell
# Randomly select ccls for each drug and visualize the drug embeddings extracted from the meta-learner.
python -u main_viz_drugs.py

# Randomly select ccls regard to drug_45 and tissue_7 for fine-tuning the model and extract the attention scores and weights.
python -u main_viz_model.py --drug_id 45 --tissue_id 7 --k 10 --rnd_seed 0

# Using CCL samples to fine-tune the model and explain TCGA samples regard to drug_9 and tissue_5.
python -u main_vizcrox_model.py --drug_id 9 --tissue_id 5 --drug_dataset_name panTCGA --ccl_dataset_name panTCGA --omics_combination E --sample_name Sample_ID 
```

