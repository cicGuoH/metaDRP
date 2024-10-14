# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:38:28 2023

@author: GUI
"""
import numpy as np
import pandas as pd
import os.path as osp
import pickle, os
import argparse
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold
from scipy import stats
from data_utils.dataset import DRDataloderForFewShotLearning

_rnd_seed = 4132231

score_func = metrics.make_scorer(metrics.mean_absolute_error)

def gridcv_svr(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=_rnd_seed)
    tuned_parameters = {'C': [2 ** j for j in range(-10, 10, 2)], "kernel":["linear", "rbf"]}
    svr_gs = GridSearchCV(SVR(), param_grid=tuned_parameters, 
                         scoring=score_func, cv=kf, n_jobs=-1)
    svr_gs.fit(X, y)
    return svr_gs.best_params_

def gridcv_rf(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=_rnd_seed)
    tuned_parameters = {'n_estimators': [500, 1000, 1500],
                       'max_features': [i for i in range(5, 2*int(np.sqrt(X.shape[1])), 500)]}
    rf_gs = GridSearchCV(RandomForestRegressor(random_state=_rnd_seed), param_grid=tuned_parameters, 
                          scoring=score_func, cv=kf, n_jobs=-1)
    rf_gs.fit(X, y)
    return rf_gs.best_params_

def gridcv_ridge(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=_rnd_seed)
    tuned_parameters = {'alpha': np.linspace(0, 1, 21)}
    lasso_gs = GridSearchCV(Ridge(random_state=_rnd_seed), param_grid=tuned_parameters, 
                          scoring=score_func, cv=kf, n_jobs=-1)
    lasso_gs.fit(X, y)
    return lasso_gs.best_params_

def get_model(model_name, X_train, y_train):
    
    if model_name == "SVR":
        params = gridcv_svr(X_train, y_train)
        model = SVR(**params).fit(X_train, y_train)
    
    elif model_name == "Ridge":
        params = gridcv_ridge(X_train, y_train)
        model = Ridge(random_state=_rnd_seed, **params).fit(X_train, y_train)
    
    elif model_name == "RF":
        params = gridcv_rf(X_train, y_train)
        model = RandomForestRegressor(random_state=_rnd_seed, **params).fit(X_train, y_train)
    return model


def model_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics_report = {}
    if len(y_pred) >= 2:
        metrics_report["PCC"] = np.corrcoef(y_test, y_pred)[0,1]
        metrics_report["R2"] = metrics.r2_score(y_test, y_pred)
        metrics_report["MAE"] = metrics.mean_absolute_error(y_test, y_pred)
        metrics_report["MSE"] = metrics.mean_squared_error(y_test, y_pred)
        metrics_report["SCC"] = stats.spearmanr(y_test, y_pred)[0]
    else:
        metrics_report["PCC"] = np.nan
        metrics_report["MAE"] = np.nan
        metrics_report["MSE"] = np.nan
        metrics_report["R2"] = np.nan
        metrics_report["SCC"] = np.nan
    return y_pred, metrics_report
    

def model_save(save_path, model_name, model):
    with open(save_path +'/%s.pickle' % model_name, "wb") as file_obj:
        pickle.dump(model,file_obj) 
    # print("Model saving is finished.")
    return False

def arg_parse():
    parser = argparse.ArgumentParser()
    
    # dataset
    parser.add_argument('--drug_dataset_name', type=str, default="GDSC1", help="directory path of drug")
    parser.add_argument('--ccl_dataset_name', type=str, default="CCL", help="directory path of ccl")
    parser.add_argument('--omics_combination', type=str, default="E", help="omics combination, etc. ECDP")
    parser.add_argument('--sample_name', type=str, default="DepMap_ID")
    parser.add_argument('--measurement_method', type=str, default="IC50")
    parser.add_argument('--dataset_name_train', type=str, default="maml_train")
    parser.add_argument('--dataset_name_val', type=str, default="maml_val")
    
    parser.add_argument('--train_seed', type=int, default=0, help="random seed of sampling train dataset")
    parser.add_argument('--val_seed', type=int, default=0, help="random seed of sampling val/test dataset")
    parser.add_argument('--num_support_samples', type=int, default=10, help="number of support ccls per task")
    parser.add_argument('--num_query_samples', type=int, default=20, help="number of query ccls per task")
    parser.add_argument('--k', type=int, default=10, help="number of support ccls per task")
    parser.add_argument('--val_samples_per_batch', type=int, default=64, help="number of query ccls per task")
    parser.add_argument('--num_batches_per_iter', type=int, default=10, help="number of tasks per iter")
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--ccl_graph', type=bool, default=False, help="input type of ccl samples, whether owns edge_index or not")
    parser.add_argument('--mini_batch', type=bool, default=True, help="whether to use mini-batch way to evaluate on a drug-tissue dataset, suitable when size is big")
    parser.add_argument('--num_pathways', type=int, default=186)
    # train
    parser.add_argument('--model_name', type=str, default="Ridge", help="model name, etc. metaDRP")
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--total_epochs', type=int, default=1)
    return parser.parse_args()

def ml_test(args):
    
    dataloader = DRDataloderForFewShotLearning(args)
    model_dir = "../model_logging/save_ml_models/%s_%s_k%d" % (args.dataset_name_val, args.model_name, args.k)
    
        
    # ML方法仅使用EXP数据
    with tqdm(total=dataloader.dataset_val.data_length) as qbar:
        for iter_count, iter_data in enumerate(dataloader.get_val_batches()):
            
            X_train = iter_data[1][0].select(dim=-1, index=0) if len(iter_data[1][0].size())==3 else iter_data[1][0]
            X_train = X_train.numpy().astype(np.float32)
            y_train = iter_data[2][0].numpy().astype(np.float32)
            
            mini_batch_length = len(iter_data[3][0])
            mini_batch_X = []
            mini_batch_y = []
            for i in range(mini_batch_length):
                query_X = iter_data[4][0][i].select(dim=-1, index=0) if len(iter_data[4][0][i].size())==3 else iter_data[4][0][i]
                query_y = iter_data[5][0][i]
                mini_batch_X.append(query_X.numpy())
                mini_batch_y.extend(query_y.tolist())
            X_val = np.concatenate(mini_batch_X, axis=0, dtype=np.float32)
            y_val = np.array(mini_batch_y, dtype=np.float32)
            
            drug_id, tissue_id, _, val_df = dataloader.dataset_val.get_sample_df(iter_count)
            # if osp.exists(osp.join(model_dir, "%s-%s" % (drug_id, tissue_id), '/%s.pickle' % args.model_name)):
            #     qbar.update(1)
            #     continue
            model = get_model(args.model_name, X_train, y_train)
            y_pred, metrics_report = model_evaluate(model, X_val, y_val)
            val_df["y_pred"] = y_pred
            metrics_report["Drug_ID"] = drug_id
            metrics_report["Tissue_ID"] = tissue_id
            metrics_report_df = pd.DataFrame(metrics_report, index=[0])    
            
            if iter_count == 0:
                metrics_report_df_all = metrics_report_df
                val_all_df = val_df
            else:
                metrics_report_df_all = pd.concat([metrics_report_df_all, metrics_report_df], axis=0, ignore_index=True)
                val_all_df = pd.concat([val_all_df, val_df], axis=0)
            
            model_save_path = osp.join(model_dir, "%s-%s" % (drug_id, tissue_id))
            if not osp.exists(model_save_path):
                os.makedirs(model_save_path)
            model_save(model_save_path, args.model_name, model)
            qbar.set_description(desc="task %d" % iter_count)
            qbar.set_postfix(PCC="%.4f" % metrics_report["PCC"], R2="%.4f" % metrics_report["R2"])
            qbar.update(1)
    
    
    val_all_df.to_csv(osp.join(model_dir, "preds.csv"))
    metrics_report_df_all.to_csv(osp.join(model_dir, "metrics_report.csv"))    
    for name in metrics_report_df_all.columns[:-2]:
        print("%s = %.4f ± %.4f" % (name, np.mean(metrics_report_df_all[name]), 1.96*np.std(metrics_report_df_all[name])/np.sqrt(len(metrics_report_df_all[name]))))

def main():
    args = arg_parse()
    args.drug_data_dir = osp.join("../data_processed", args.drug_dataset_name)
    args.dr_data_dir = osp.join("../data_processed", args.drug_dataset_name, "%s_%s" % (args.drug_dataset_name, args.omics_combination))
    args.ccl_data_path = osp.join("../data_processed", args.ccl_dataset_name,"%s_%s_p%d.npy" % (args.ccl_dataset_name, args.omics_combination, args.num_pathways))
    
    for model_name in ["RF", "Ridge", "SVR"]:
        for k in range(5, 16):
            args.model_name = model_name
            args.k = k
            ml_test(args)
            
if __name__ == "__main__":
    main()
        
        
        
        