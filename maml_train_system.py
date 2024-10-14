# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:05:32 2023

@author: GUI
"""

import torch
import torch.nn as nn
import os, typing
from tqdm import tqdm
import numpy as np
import pandas as pd
import higher
import random
import torchmetrics.functional as tmf
from torch.utils.tensorboard import SummaryWriter
from models.meta_model_architectures import metaDRP
from model_utils.lr_schedular import build_scheduler
# from model_utils.get_optimizer import LSLRGradientDescentLearningRule as LSLR

_random_seed = 4132231

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


class MAMLBase:
    
    def __init__(self, args):
        
        self.args = args
        self.loss_function = get_loss_fn("regression")
        self.saved_model_path, self.log_path, self.visual_path = self.build_task_folder(self.args.task_name)
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
            
    def load_model(self, resume_epoch=0, num_iters_per_epoch=None, model_name="metaDRP"):
        
        model = dict.fromkeys(["hyper_net", "f_base_net", "optimizer", "lr_scheduler"])
        
        if model_name == "metaDRP":
            base_net = metaDRP(args=self.args, ccl_drug_trans=self.args.ccl_drug_trans)
        
        if self.args.pretrain_flag:
            drug_ckpt_path = os.path.join(self.args.drug_pretrain_path, "saved_model", 'best_model.pt')
            # ccl_ckpt_path = os.path.join(self.args.ccl_pretrain_path, "saved_model", 'best_model.pt')
            drug_saved_ckpt = torch.load(
                f=drug_ckpt_path,
                map_location=self.args.device)
            # ccl_saved_ckpt = torch.load(
            #     f=ccl_ckpt_path,
            #     map_location=self.args.device)
            
            base_net.drug_enc.load_state_dict(drug_saved_ckpt["graphencoder"])
            # base_net.ccl_enc.load_state_dict(ccl_saved_ckpt["ccl_enc"])
        params = torch.nn.utils.parameters_to_vector(parameters=base_net.parameters())
        print('Number of parameters of the base network = {0:,}.\n'.format(params.numel()))
        
        model["hyper_net"] = IdentityNet(base_net)
        
        # move to device
        model["hyper_net"].to(self.args.device)
        base_net.to(self.args.device)
        model["f_base_net"] = self.torch_module_to_functional(torch_net=base_net)
        if self.args.mode=="train":
            # optimizer
            model["optimizer"] = torch.optim.Adam(params=model["hyper_net"].parameters(), lr=self.args.meta_lr)
            # model["inner_optimizer"] = LSLR(device=self.args.device, 
            #                                 num_inner_updates=self.args.num_inner_updates, 
            #                                 use_learnable_learning_rates=self.args.use_learnable_learning_rates,
            #                                 init_learning_rate=self.args.inner_lr)
            # model["inner_optimizer"].initialise(model["hyper_net"].forward())
            model["lr_scheduler"] = build_scheduler(self.args, optimizer=model["optimizer"], n_iter_per_epoch=num_iters_per_epoch)
            
            # load model if resume_epoch>0
            if resume_epoch > 0:
                # ckpt_path
                ckpt_path = os.path.join(self.saved_model_path, 'Epoch_{0:d}.pt'.format(resume_epoch))
                
                saved_ckpt = torch.load(
                    f=ckpt_path,
                    map_location=self.args.device)
                
                # load state dictionaries
                model["hyper_net"].load_state_dict(state_dict=saved_ckpt['hyper_net_state_dict'])
                model["optimizer"].load_state_dict(state_dict=saved_ckpt['opt_state_dict'])
                # model["inner_optimizer"].load_state_dict(state_dict=saved_ckpt['inner_opt_state_dict'])
                model["lr_scheduler"].load_state_dict(state_dict=saved_ckpt['sche_state_dict'])
                # # update learning rate
                # for param_group in model["optimizer"].param_groups:
                #     if param_group['lr'] != self.args.meta_lr:
                #         param_group['lr'] = self.args.meta_lr
                
        elif self.args.mode=="test":
            # ckpt_path
            ckpt_path = os.path.join(self.saved_model_path, 'best_model.pt')
            saved_ckpt = torch.load(
                f=ckpt_path,
                map_location=self.args.device)
            # load state dictionaries
            model["hyper_net"].load_state_dict(state_dict=saved_ckpt['hyper_net_state_dict'])
        return model
    
    
    def prediction(self, x, adapted_hyper_net, model:dict):
        base_net_params = adapted_hyper_net.forward()
        logits = model["f_base_net"].forward(x, params=base_net_params)
        return logits
    
                    
    def inner_loop(self, task_x, task_y, model:dict):
        """
        Task adaptation step that produces a task-specific model
        Args:
            x: training data of a task
            y: training labels of that task
            model: a dictionary consisting of "hyper_net", "f_base_net", "optimizer" 
            
        Returns:
            a task-specific model
        """
        # 将hyper_net转变为functional模式，可以进行传参
        f_hyper_net = higher.patch.monkeypatch(
           module=model["hyper_net"],
           copy_initial_weights=False,
           track_higher_grads=True if self.args.mode=="train" else False)
        
        support_loss_innerloop = []
        for inner_step in range(self.args.num_inner_updates):
            # parameters of the task-specific hyper_net
            q_params = f_hyper_net.fast_params
            
            # generate task-specific parameter
            base_net_params = f_hyper_net.forward()
            
            # predict output logits
            logits = model["f_base_net"].forward(task_x, params=base_net_params)
            
            # calculate loss
            loss = self.loss_function(input=logits, target=task_y)
            support_loss_innerloop.append(loss.item())
            
            # calculate grads
            grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=q_params,
                    create_graph=not self.args.first_order,
                    allow_unused=True
                    )
            
            # get new params
            # new_q_params = model["inner_optimizer"].update_params(q_params, grads, num_step=inner_step)
            # f_hyper_net.update_params(new_q_params)
            new_q_params = []
            for param, grad in zip(q_params, grads):
                if param.requires_grad:
                    new_q_params.append(higher.optim._add(tensor=param, a1=-self.args.inner_lr, a2=grad))
                else:
                    new_q_params.append(param)
            f_hyper_net.update_params(new_q_params)
        return f_hyper_net, support_loss_innerloop
        
    
    def outer_loop(self, batch_data, model:dict):
        """
        for a batch, use task_average_loss to update base_net
        """
        loss_query_batch = []
        loss_support_batch = []
        
        for task_id in range(self.args.num_batches_per_iter):
            
            support_drug_fea = batch_data[0][task_id].to(self.args.device)
            support_ccl_fea = batch_data[1][task_id].to(self.args.device)
            support_y = batch_data[2][task_id].to(self.args.device)
            query_drug_fea = batch_data[3][task_id].to(self.args.device)
            query_ccl_fea = batch_data[4][task_id].to(self.args.device)
            query_y = batch_data[5][task_id].to(self.args.device)
            
            # inner_loop
            adapted_hyper_net, loss_support_task = self.inner_loop(task_x=(support_drug_fea, support_ccl_fea), task_y=support_y, model=model)
            
            # validation on query data
            logits_query = self.prediction(x=(query_drug_fea, query_ccl_fea), adapted_hyper_net=adapted_hyper_net, model=model)
            loss_query_task = self.loss_function(input=logits_query, target=query_y)
            loss_query_batch.append(loss_query_task)
            loss_support_batch.append(loss_support_task)
            if torch.isnan(loss_query_task):
                raise ValueError("Loss is NaN.")
        mean_loss_query_batch = torch.stack(loss_query_batch).mean() 
        mean_loss_support_batch = np.mean(np.array(loss_support_batch), axis=0) # for writer
        
        # calculate gradients w.r.t. hyper_net's parameters
        return mean_loss_query_batch, mean_loss_support_batch
    
    
    def train(self, dataloader, valid=True):
        num_iters_per_epoch = dataloader.num_iters_per_epoch
        # initialize or load model
        model = self.load_model(resume_epoch=self.args.resume_epoch, num_iters_per_epoch=num_iters_per_epoch, model_name=self.args.model_name)
        model["optimizer"].zero_grad()
        
        
        # logging
        tb_writer = SummaryWriter(
            log_dir=self.log_path,
            purge_step=self.args.resume_epoch * num_iters_per_epoch if self.args.resume_epoch > 0 else None
        )
        try:
            best_val_loss = 9999.
            patience = self.args.patience
            best_epoch = self.args.resume_epoch
            for epoch_id in range(self.args.resume_epoch, self.args.resume_epoch+self.args.total_epochs, 1):
                if (epoch_id - best_epoch) > patience:
                    break
                loss_monitor = {}
                loss_monitor["loss_query"] = 0.
                loss_monitor["loss_support"] = []
                with tqdm(total=num_iters_per_epoch, desc="Epoch %d" % (epoch_id+1)) as qbar:
                    for iter_count, iter_data in enumerate(dataloader.get_train_batches()):
                        current_iter = epoch_id * num_iters_per_epoch + iter_count + 1
                        mean_loss_query_batch, mean_loss_support_batch = self.outer_loop(batch_data=iter_data, model=model)
                        
                        model["optimizer"].zero_grad()
                        mean_loss_query_batch.backward()
                        if self.args.clip_grad > 0.:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
                        # update meta-parameters
                        model["optimizer"].step()
                        model["lr_scheduler"].step_update(current_iter)
                        
                        # monitor
                        loss_monitor["loss_query"] = mean_loss_query_batch.item()
                        loss_monitor["loss_support"] =  mean_loss_support_batch
                        
                        tb_writer.add_scalar(tag="Train_loss_query", scalar_value=loss_monitor["loss_query"], global_step=current_iter)
                        tb_writer.add_scalar(tag="learning_rate", scalar_value=model["optimizer"].state_dict()['param_groups'][0]['lr'], global_step=current_iter)
                        for iter_inner in range(len(loss_monitor["loss_support"])):
                            tb_writer.add_scalar(tag="Train_loss_support_%d"%iter_inner, scalar_value=loss_monitor["loss_support"][iter_inner], global_step=current_iter)
                        
                        # reset monitoring variables
                        loss_monitor["loss_query"] = 0.
                        loss_monitor["loss_support"] = []
                        qbar.update(1)
                        
                    # validation per epoch
                    if valid:
                        model["f_base_net"].eval()
                        metrics_report, _ = self.evaluate(dataloader, model)
                        for key in metrics_report.keys():
                            if key not in ["Drug_ID", "Tissue_ID"]:
                                tb_writer.add_scalar(tag="Val_%s"%key, scalar_value=np.mean(metrics_report[key]), global_step=current_iter)
                        qbar.write(str({key: "%.4f ± %.4f" % (np.mean(metrics_report[key]), 1.96*np.std(metrics_report[key])/np.sqrt(len(metrics_report[key]))) for key in metrics_report if key not in ["Drug_ID", "Tissue_ID"]}))
                        model["f_base_net"].train()
                        if best_val_loss > np.mean(metrics_report["MAE"]):
                            best_val_loss = np.mean(metrics_report["MAE"])
                            best_epoch = epoch_id + 1
                            ckpt = {
                                "hyper_net_state_dict": model["hyper_net"].state_dict(),
                                "opt_state_dict": model["optimizer"].state_dict(),
                                "sche_state_dict":model["lr_scheduler"].state_dict()
                                }
                            ckpt_path = os.path.join(self.saved_model_path, 'best_model.pt')
                            torch.save(obj=ckpt, f=ckpt_path)
                            
                        del metrics_report
                        
                    # save model per epoch
                    ckpt = {
                        "hyper_net_state_dict": model["hyper_net"].state_dict(),
                        "opt_state_dict": model["optimizer"].state_dict(),
                        # "inner_opt_state_dict": model["inner_optimizer"].state_dict(),
                        "sche_state_dict":model["lr_scheduler"].state_dict()
                        }
                    ckpt_path = os.path.join(self.saved_model_path, 'Epoch_{0:d}.pt'.format(epoch_id + 1))
                    torch.save(obj=ckpt, f=ckpt_path)
                    
        finally:
            print('\nBest_epoch:%d ,Close tensorboard summary writer' % best_epoch)
            tb_writer.close()
        
        return None
            
    
    def evaluate(self, dataloader, model:dict, return_metrics=True):
        
        metrics = {}
        for iter_count, iter_data in enumerate(dataloader.get_val_batches()):
            
            support_drug_fea = iter_data[0][0].to(self.args.device)
            support_ccl_fea = iter_data[1][0].to(self.args.device)
            support_y = iter_data[2][0].to(self.args.device)
            adapted_hyper_net, _ = self.inner_loop(task_x=(support_drug_fea, support_ccl_fea), task_y=support_y, model=model)
            drug_id, tissue_id, _, query_df =  dataloader.dataset_val.get_sample_df(iter_count)
            mini_batch_length = len(iter_data[4][0])
            mini_batch_y = []
            mini_batch_logits = []
            for i in range(mini_batch_length):
                
                query_drug_fea = iter_data[3][0][i].to(self.args.device)
                query_ccl_fea = iter_data[4][0][i].to(self.args.device)
                query_y = iter_data[5][0][i].to(self.args.device)
                logits = self.prediction(x=(query_drug_fea, query_ccl_fea), adapted_hyper_net=adapted_hyper_net, model=model)
                
                if self.args.device != "cpu":
                    mini_batch_y.extend(query_y.cpu().tolist())
                    mini_batch_logits.extend(logits.cpu().tolist())
                else:
                    mini_batch_y.extend(query_y.tolist())
                    mini_batch_logits.extend(logits.tolist())
            query_df["logits"] = mini_batch_logits
            mini_batch_logits = torch.tensor(mini_batch_logits)
            mini_batch_y = torch.tensor(mini_batch_y)
            
            if return_metrics:
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
            
        if return_metrics:
            return metrics, query_df_all
        else:
            return  query_df_all
    
    
    def test(self, dataloader, return_metrics=True):
        print("Start evaluation on test dataset...")
        model = self.load_model(num_iters_per_epoch=1)
        if return_metrics:
            metrics, query_df_all = self.evaluate(dataloader, model=model)
            return metrics, query_df_all
        else:
            query_df_all = self.evaluate(dataloader, model=model, return_metrics=False)
            return query_df_all
    
    def evaluate_zero_shot(self, dataloader, model:dict, return_metrics=True):
       
        metrics = {}
        for iter_count, iter_data in enumerate(dataloader.get_val_batches()):
            
            adapted_hyper_net = higher.patch.monkeypatch(module=model["hyper_net"],
                                                         copy_initial_weights=False,
                                                         track_higher_grads=True if self.args.mode=="train" else False)
            drug_id, tissue_id, query_df = dataloader.dataset_val.get_sample_df(iter_count)
            mini_batch_length = len(iter_data[1][0])
            mini_batch_y = []
            mini_batch_logits = []
            for i in range(mini_batch_length):
                
                query_drug_fea = iter_data[0][0][i].to(self.args.device)
                query_ccl_fea = iter_data[1][0][i].to(self.args.device)
                query_y = iter_data[2][0][i].to(self.args.device)
                logits = self.prediction(x=(query_drug_fea, query_ccl_fea), adapted_hyper_net=adapted_hyper_net, model=model)
                
                if self.args.device != "cpu":
                    mini_batch_y.extend(query_y.cpu().tolist())
                    mini_batch_logits.extend(logits.cpu().tolist())
                else:
                    mini_batch_y.extend(query_y.tolist())
                    mini_batch_logits.extend(logits.tolist())
            query_df["logits"] = mini_batch_logits        
            mini_batch_logits = torch.tensor(mini_batch_logits)
            mini_batch_y = torch.tensor(mini_batch_y)
            
            if return_metrics:
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
        if return_metrics:        
            return metrics, query_df_all
        else:
            return query_df_all
        
            
    def test_zero_shot(self, dataloader, return_metrics=True):
        print("Start evaluation on test dataset...")
        model = self.load_model(num_iters_per_epoch=1)
        if return_metrics:
            metrics, query_df_all = self.evaluate_zero_shot(dataloader, model=model)
            return metrics, query_df_all
        else:
            query_df_all = self.evaluate_zero_shot(dataloader, model=model, return_metrics=False)
            return query_df_all
    
    @staticmethod
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
        
        
    @staticmethod
    def torch_module_to_functional(torch_net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
        """Convert a conventional torch module to its "functional" form
        """
        f_net = higher.patch.make_functional(module=torch_net)
        higher.patch.buffer_sync(module=torch_net, fmodule=f_net)
        f_net.track_higher_grads = False
        f_net._fast_params = [[]]
        return f_net
    
    
