# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:49:07 2023

@author: GUI
"""

import torch
import torch.nn as nn
import os, time, typing
import os.path as osp
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import argparse
import torchmetrics.functional as tmf
from torch.utils.tensorboard import SummaryWriter
from models.meta_model_architectures import metaDRP
from model_utils.lr_schedular import build_scheduler
from data_utils.dataset import DRDataloderForTraditionalTraining
import higher

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
        
def build_task_folder(task_name):
    par_path = os.path.abspath(os.path.dirname(os.getcwd()))
    task_path = os.path.join(par_path, "model_logging",  task_name)
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

class IdentityNet(torch.nn.Module):
    """Identity hyper-net class for MAML"""
    def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
        super(IdentityNet, self).__init__()
        base_state_dict = base_net.state_dict()

        params = list(base_state_dict.values())
        # params = intialize_parameters(state_dict=base_state_dict)

        self.params = torch.nn.ParameterList([torch.nn.Parameter(p.float()) \
            for p in params])
        self.identity = torch.nn.Identity()

    def forward(self) -> typing.List[torch.Tensor]:
        out = []
        for param in self.params:
            temp = self.identity(param)
            out.append(temp)
        return out

def get_loss_fn(task="regression"):
    
    if task == "regression":
        criterion = nn.L1Loss(reduction="mean")
    
    elif task == "classification":
        criterion = nn.CrossEntropyLoss(size_average=True)
        
    return criterion

def report_metrics(logits, target, task="regression"):
    metrics = {}
    if task == "regression":
        if len(logits) >= 2:
            pcc = tmf.regression.concordance_corrcoef(preds=logits, target=target)
            mae = tmf.regression.mean_absolute_error(preds=logits, target=target)
            mse = tmf.regression.mean_squared_error(preds=logits, target=target)
            r2 = tmf.regression.r2_score(preds=logits, target=target)
            scc = tmf.regression.spearman_corrcoef(preds=logits, target=target)
            metrics["PCC"] = pcc.item()
            metrics["MAE"] = mae.item()
            metrics["MSE"] = mse.item()
            metrics["R2"] = r2.item()
            metrics["SCC"] = scc.item()
        else:
            metrics["PCC"] = np.nan
            metrics["MAE"] = np.nan
            metrics["MSE"] = np.nan
            metrics["R2"] = np.nan
            metrics["SCC"] = np.nan
    elif task == "classification":
        raise NotImplementedError()
    return metrics

class TraditionalBase:
    
    def __init__(self, args):
        
        self.args = args
        self.loss_function = get_loss_fn("regression")
        self.saved_model_path, self.log_path, self.visual_path = build_task_folder(self.args.task_name)
        self.set_random_seed(seed=_random_seed)
        
    def set_random_seed(self, seed, deterministic=True):
        """Set random seed."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def load_model(self, resume_epoch, num_iters_per_epoch=None, model_name="DRP"):
        model = dict.fromkeys(["base_net", "optimizer", "lr_scheduler"])
        base_net = metaDRP(args=self.args, ccl_drug_trans=self.args.ccl_drug_trans)
        
        if self.args.pretrain_flag:
            drug_ckpt_path = os.path.join(self.args.drug_pretrain_path, "saved_model", 'best_model.pt')
            drug_saved_ckpt = torch.load(
                f=drug_ckpt_path,
                map_location=self.args.device)
            base_net.drug_enc.load_state_dict(drug_saved_ckpt["graphencoder"])
            
        params = torch.nn.utils.parameters_to_vector(parameters=base_net.parameters())
        print('Number of parameters of the base network = {0:,}.\n'.format(params.numel()))
        
        
        # move to device
        base_net.to(self.args.device)
        model["base_net"] = base_net
        # optimizer
        model["optimizer"] = torch.optim.Adam(params=base_net.parameters(), lr=self.args.lr)
        model["lr_scheduler"] = build_scheduler(self.args, optimizer=model["optimizer"], n_iter_per_epoch=num_iters_per_epoch)
        
        # load model if resume_epoch>0
        if resume_epoch > 0 and self.args.mode=="train":
            # ckpt_path
            ckpt_path = os.path.join(self.saved_model_path, 'Epoch_{0:d}.pt'.format(resume_epoch))
            
            saved_ckpt = torch.load(
                f=ckpt_path,
                map_location=self.args.device)
            
            # load state dictionaries
            model["base_net"].load_state_dict(state_dict=saved_ckpt['base_net_state_dict'])
            model["optimizer"].load_state_dict(state_dict=saved_ckpt['opt_state_dict'])
            model["lr_scheduler"].load_state_dict(state_dict=saved_ckpt['sche_state_dict'])
            
        elif self.args.mode=="test":
            # ckpt_path
            ckpt_path = os.path.join(self.saved_model_path, 'best_model.pt')
            saved_ckpt = torch.load(
                f=ckpt_path,
                map_location=self.args.device)
            # load state dictionaries
            model["base_net"].load_state_dict(state_dict=saved_ckpt['base_net_state_dict'])
        return model
  
    def train(self, dataloader):
        loss_fn = get_loss_fn(task="regression")
        saved_model_path, log_path, visual_path = build_task_folder(self.args.task_name)
        best_epoch = self.args.resume_epoch
        best_loss = 9999.
        patience = 50
        # load model
        num_iters_per_epoch = dataloader.num_iters_per_epoch
        model = self.load_model(resume_epoch=self.args.resume_epoch, num_iters_per_epoch=num_iters_per_epoch, model_name=self.args.model_name)
        
        
        # logging
        tb_writer = SummaryWriter(
            log_dir=self.log_path,
            purge_step=self.args.resume_epoch * num_iters_per_epoch if self.args.resume_epoch > 0 else None
        )
        
        try:
            
            for epoch_id in range(self.args.resume_epoch, self.args.resume_epoch+self.args.total_epochs, 1):
                if (epoch_id - best_epoch) > patience:
                    break
                with tqdm(total=num_iters_per_epoch, desc="Epoch %d" % (epoch_id+1)) as qbar:
                    for iter_count, iter_data in enumerate(dataloader.get_train_batches()):
                        current_iter = epoch_id * num_iters_per_epoch + iter_count + 1
                        drug_fea, ccl_fea, y = iter_data
                        
                        # confirm device
                        drug_fea = drug_fea.to(self.args.device)
                        ccl_fea = ccl_fea.to(self.args.device)
                        y = y.to(self.args.device)
                        logits = model["base_net"].forward((drug_fea, ccl_fea))
                        train_loss = loss_fn(logits, y)
                        
                        model["optimizer"].zero_grad()
                        train_loss.backward()
                        model["optimizer"].step()
                        model["lr_scheduler"].step_update(current_iter)
                        qbar.update(1)
                
                tb_writer.add_scalar(tag="Train_loss", scalar_value=train_loss, global_step=current_iter)
                # valid
                model["base_net"].eval()
                y_true = []
                y_pred = []
                
                with torch.no_grad():
                    for batch_data_val in dataloader.get_val_batches():
                        drug_fea, ccl_fea, y = batch_data_val
                        drug_fea = drug_fea.to(self.args.device)
                        ccl_fea = ccl_fea.to(self.args.device)
                        logits = model["base_net"].forward((drug_fea, ccl_fea))
                        
                        y_true.extend(y.cpu().tolist())
                        y_pred.extend(logits.cpu().tolist())
                    y_true = torch.tensor(y_true)
                    y_pred = torch.tensor(y_pred)
                    metrics_report = report_metrics(logits=y_pred, target=y_true)
                
                qbar.write(str({key: "%.4f" % metrics_report[key] for key in metrics_report}))
                tb_writer.add_scalar(tag="Val_loss", scalar_value=metrics_report["MAE"], global_step=epoch_id)
                
                ckpt = {
                    "base_net_state_dict": model["base_net"].state_dict(),
                    "opt_state_dict": model["optimizer"].state_dict(),
                    "sche_state_dict": model["lr_scheduler"].state_dict()
                    }
                ckpt_path = os.path.join(saved_model_path, 'best_model.pt')
                torch.save(obj=ckpt, f=ckpt_path)
                
                if metrics_report["MAE"] < best_loss:
                    best_loss = metrics_report["MAE"]
                    best_epoch = epoch_id + 1
                    
                    ckpt = {
                        "base_net_state_dict": model["base_net"].state_dict(),
                        "opt_state_dict": model["optimizer"].state_dict(),
                        "sche_state_dict": model["lr_scheduler"].state_dict()
                        }
                    ckpt_path = os.path.join(saved_model_path, 'Epoch_{0:d}.pt'.format(epoch_id + 1))
                    torch.save(obj=ckpt, f=ckpt_path)
        finally:
            print('\nBest_epoch:%d, Close tensorboard summary writer' % best_epoch)
            tb_writer.close()
        return None
    
    def test_wo_tf(self, dataloader):
        print("Start evaluation on test dataset...")
        model = self.load_model(resume_epoch=self.args.resume_epoch, num_iters_per_epoch=1)
        
        query_df = dataloader.dataset_val.dr_data
        query_logits_batch = []
        query_y_batch = []
        for iter_count, iter_data in enumerate(dataloader.get_val_batches()):
            query_drug_fea = iter_data[0].to(self.args.device)
            query_ccl_fea = iter_data[1].to(self.args.device)
            query_y = iter_data[2].to(self.args.device)
            
            # Prediction
            with torch.no_grad():
                query_logits = model["base_net"].forward((query_drug_fea, query_ccl_fea))
            if self.args.device != "cpu":
                query_logits = query_logits.cpu()
                # query_y = query_y.cpu()
            query_logits_batch.extend(query_logits.tolist())
            query_y_batch.extend(query_y.tolist())
        query_df["logits"] = query_logits_batch
        return query_df
        
    def test_w_tf(self, dataloader):
        print("Start evaluation on test dataset...")
        model = self.load_model(resume_epoch=self.args.resume_epoch, num_iters_per_epoch=1)
        metrics = {}
        hyper_net = IdentityNet(model["base_net"])
        f_base_net = self.torch_module_to_functional(torch_net=model["base_net"])
        
        for iter_count, iter_data in enumerate(dataloader.get_val_batches()):
                
            ft_drug_fea = iter_data[0][0].to(self.args.device)
            ft_ccl_fea = iter_data[1][0].to(self.args.device)
            ft_y = iter_data[2][0].to(self.args.device)
            
            # Finetune with setting steps and fixed learning-rate
            f_hyper_net = higher.patch.monkeypatch(
               module=hyper_net,
               copy_initial_weights=False,
               track_higher_grads=False)
            
            for ft_step in range(self.args.ft_steps):
                
                # parameters of the task-specific hyper_net
                q_params = f_hyper_net.fast_params
                
                # generate task-specific parameter
                base_net_params = f_hyper_net.forward()
                
                # predict output logits
                logits = f_base_net.forward(((ft_drug_fea, ft_ccl_fea)), params=base_net_params)
                loss = self.loss_function(input=logits, target=ft_y)
                
                # calculate grads
                grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=q_params,
                        create_graph=not self.args.first_order,
                        allow_unused=True
                        )
                
                new_q_params = []
                for param, grad in zip(q_params, grads):
                    if param.requires_grad:
                        new_q_params.append(higher.optim._add(tensor=param, a1=-self.args.ft_lr, a2=grad))
                    else:
                        new_q_params.append(param)
                f_hyper_net.update_params(new_q_params)
            
            
            # Prediction
            with torch.no_grad():
                base_net_params = f_hyper_net.forward()
                drug_id, tissue_id, _, query_df =  dataloader.dataset_val.get_sample_df(iter_count)
                mini_batch_length = len(iter_data[4][0])
                mini_batch_y = []
                mini_batch_logits = []
                for i in range(mini_batch_length):
                    query_drug_fea = iter_data[3][0][i].to(self.args.device)
                    query_ccl_fea = iter_data[4][0][i].to(self.args.device)
                    query_y = iter_data[5][0][i].to(self.args.device)
                    
                    query_logits = f_base_net.forward((query_drug_fea, query_ccl_fea), params=base_net_params)
                    
                    if self.args.device != "cpu":
                        mini_batch_y.extend(query_y.cpu().tolist())
                        mini_batch_logits.extend(query_logits.cpu().tolist())
                    
                    else:
                        mini_batch_y.extend(query_y.tolist())
                        mini_batch_logits.extend(query_logits.tolist())
                
            query_df["logits"] = mini_batch_logits        
            mini_batch_logits = torch.tensor(mini_batch_logits)
            mini_batch_y = torch.tensor(mini_batch_y)
            query_df["Tissue_ID"] = tissue_id
            metrics_iter = report_metrics(logits=mini_batch_logits, target=mini_batch_y)
            metrics_iter["Drug_ID"] = drug_id
            metrics_iter["Tissue_ID"] = tissue_id        
            
            for key in metrics_iter:
                if key not in metrics.keys():
                    metrics[key] = []
                else:
                    metrics[key].append(metrics_iter[key])
             
            if iter_count == 0:
                query_df_all = query_df.copy()
            else:
                query_df_all = pd.concat([query_df_all, query_df], axis=0, ignore_index=True)
        return metrics, query_df_all
    
    
    @staticmethod
    def torch_module_to_functional(torch_net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
        """Convert a conventional torch module to its "functional" form
        """
        f_net = higher.patch.make_functional(module=torch_net)
        higher.patch.buffer_sync(module=torch_net, fmodule=f_net)
        f_net.track_higher_grads = False
        f_net._fast_params = [[]]
        return f_net
