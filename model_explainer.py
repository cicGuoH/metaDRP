# -*- coding: utf-8 -*-
"""
解释器的功能
1. 提供单个样本对的attn、embeddings
2. 根据数据的情况分为 explain和crox_explain两种情况
3. 为了方便，集成了交叉预测的功能（原本的预测功能在
"""


import torch
import torch.nn as nn
import os, typing
import os.path as osp
import numpy as np
import pandas as pd
import higher
import random
import torchmetrics.functional as tmf
from models.meta_model_architectures import metaDRP
from data_utils.collator import collator_forDrug

    

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
        

def model_add_hooks1(model, model_hooks):
        
    for n, m in model.named_modules():
        if n.split(".")[-1] =="self_attn": #drug
            def hooktool1(module, fea_in, fea_out):
                model_hooks["drug_attn"].append(fea_out[-1].detach().cpu())
            m.register_forward_hook(hooktool1)
        
        elif n.split(".")[-1] =="mhAttn": #pathway
            def hooktool2(module, fea_in, fea_out):
                model_hooks["pathway_attn"].append(fea_out[-1].detach().cpu())
            m.register_forward_hook(hooktool2)
            
        elif n == "CDTrans.dropout_module": #cd
            def hooktool3(module, fea_in, fea_out):
                model_hooks["cd_attn"].append(fea_in[0].detach().cpu())
            m.register_forward_hook(hooktool3)
            
        elif n=="ccl_enc.splinear":
            def hooktool6(module, fea_in, fea_out):
                model_hooks["ccl_weights"] = module.weights.detach().cpu()
            m.register_forward_hook(hooktool6)

def model_add_hooks2(model, model_hooks):
        
    for n, m in model.named_modules():
        if n=="ffn.drop":
            def hooktool1(module, fea_in, fea_out):
                model_hooks["cd_embed"] = fea_out[0].detach().cpu()
            m.register_forward_hook(hooktool1)
            
        elif n=="drug_enc.layers.11.final_layer_norm":
            def hooktool2(module, fea_in, fea_out):
                model_hooks["d_embed"] = fea_out[0].detach().cpu()
            m.register_forward_hook(hooktool2)
        

class ModelExplainer:
    
    def __init__(self, args):
        
        self.args = args
        self.loss_function = get_loss_fn("regression")
        self.drug_data_dir = args.drug_data_dir
        self.dr_data_dir = args.dr_data_dir
        self.ccl_data_path = args.ccl_data_path
        self.sample_name = args.sample_name
        self.saved_model_path, _, _ = self.build_task_folder(self.args.task_name)
        self.src_drug_data_dir = osp.join("../data_processed", "GDSC1")
        self.src_ccl_data_path = osp.join("../data_processed","CCL","%s_%s_p%d.npy" % ("CCL", args.omics_combination, args.num_pathways))
        self.src_dr_data_dir = osp.join("../data_processed", "GDSC1", "%s_%s" % ("GDSC1", args.omics_combination))
        self.meta_model = self.load_model(model_name=args.model_name)
        print("Finish loading well-trained meta-Learner.")
        
    def load_model(self, model_name="metaDRP"):
        
        model = dict.fromkeys(["hyper_net", "f_base_net"])
        
        if model_name == "metaDRP":
            base_net = metaDRP(args=self.args, ccl_drug_trans=self.args.ccl_drug_trans)
            # for n, _ in base_net.named_modules():
            #     print(n)
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
            new_q_params = []
            for param, grad in zip(q_params, grads):
                if param.requires_grad:
                    new_q_params.append(higher.optim._add(tensor=param, a1=-self.args.inner_lr, a2=grad))
                else:
                    new_q_params.append(param)
            f_hyper_net.update_params(new_q_params)
        return f_hyper_net, support_loss_innerloop
    
    
    def get_support_samples(self, drug_id, tissue_id, k, tar_ccls, rnd_seed):
        # 输入药物ID，组织ID，以及要获得的support samples的数量k
        # 输出support samples
        drug_dict = np.load(osp.join(self.drug_data_dir, "drug_dict.npy"), allow_pickle=True).item()
        ccl_dict = np.load(osp.join(self.ccl_data_path), allow_pickle=True).item()
        drug_subset = pd.read_csv(osp.join(self.dr_data_dir, drug_id, "%s-%s.csv"%(drug_id, tissue_id)))
        
        # 排除需要解释的样本
        drug_subset_filter = drug_subset[~drug_subset[self.sample_name].isin(tar_ccls)]
        if len(drug_subset_filter) < k:
            print("not enough support samples. try k≤%d or decrease input sample size" % (len(drug_subset)-len(tar_ccls)))
            return None
        
        rng = np.random.RandomState(rnd_seed)
        sample_idx = rng.choice([i for i in range(len(drug_subset_filter))], k)
        support_df = drug_subset_filter.iloc[sample_idx, :]
        
        drug_fea = drug_dict[drug_id]
        support_drug_fea = collator_forDrug([drug_fea for i in range(k)])
        support_ccl_fea = [ccl_dict[ccl_item] for ccl_item in support_df[self.sample_name]]
        support_ccl_fea = torch.stack(support_ccl_fea, dim=0)
        support_y = torch.tensor(support_df[self.args.measurement_method].values.tolist())
        return support_drug_fea, support_ccl_fea, support_y
    
    
    def get_target_samples(self, drug_id, tar_ccls):
        drug_dict = np.load(osp.join(self.drug_data_dir, "drug_dict.npy"), allow_pickle=True).item()
        ccl_dict = np.load(osp.join(self.ccl_data_path), allow_pickle=True).item()
        drug_fea = [drug_dict[drug_id] for i in range(len(tar_ccls))]
        ccl_fea = [ccl_dict[ccl_item] for ccl_item in tar_ccls]
        return drug_fea, ccl_fea
    
    
    # main function1
    def explain_samples(self, drug_id, tissue_id, tar_ccls, rnd_seed, k):
        
        if k > 0:
            # 模型fine-tune
            support_drug_fea, support_ccl_fea, support_y = self.get_support_samples(drug_id, tissue_id, k, tar_ccls, rnd_seed)
            f_hyper_net, _ = self.inner_loop(task_x=(support_drug_fea.to(self.args.device), support_ccl_fea.to(self.args.device)), 
                                                                  task_y=(support_y).to(self.args.device), 
                                                                  model=self.meta_model)
        else:
            f_hyper_net = higher.patch.monkeypatch(
                            module=self.meta_model["hyper_net"],
                            copy_initial_weights=False,
                            track_higher_grads=True if self.args.mode=="train" else False)
        tar_drug_fea_ls, tar_ccl_fea_ls = self.get_target_samples(drug_id, tar_ccls)
        print("Start to explain...")
        for i, tar_ccl in enumerate(tar_ccls):
            save_path = osp.join(self.args.saved_path, "%s_%s" % (drug_id, tar_ccl), "%s_%s_rm%d" % (drug_id, tar_ccl, rnd_seed))
            if not osp.exists(save_path):
                os.makedirs(save_path)
            
            model_hooks_tmp = {
                "drug_attn":[],
                "pathway_attn":[],
                "cd_attn":[],
                "ccl_weights":None,
                }
            
            model_add_hooks1(model=self.meta_model["f_base_net"], model_hooks=model_hooks_tmp)
            tar_drug_fea = collator_forDrug([tar_drug_fea_ls[i]]).to(self.args.device)
            tar_ccl_fea = tar_ccl_fea_ls[i].unsqueeze(dim=0).to(self.args.device)
            
            _ = self.prediction(x=(tar_drug_fea, tar_ccl_fea), adapted_hyper_net=f_hyper_net, model=self.meta_model)

            tmp_mat = np.stack(model_hooks_tmp["drug_attn"], axis=0)
            np.save(osp.join(save_path, "drug_attn.npy"), tmp_mat.squeeze())
            print(tmp_mat.squeeze().shape)
            
            tmp_mat = np.stack(model_hooks_tmp["pathway_attn"], axis=0)
            np.save(osp.join(save_path, "pathway_attn.npy"), tmp_mat.squeeze())
            print(tmp_mat.squeeze().shape)
            
            
            tmp_mat = np.stack(model_hooks_tmp["cd_attn"], axis=0)
            np.save(osp.join(save_path, "cd_attn.npy"), tmp_mat.squeeze())
            print(tmp_mat.squeeze().shape)
            
            np.save(osp.join(save_path, "ccl_weights.npy"), model_hooks_tmp["ccl_weights"].numpy())
            print(model_hooks_tmp["ccl_weights"].numpy().shape)
        print("Finished.")        
        return None
    
    # main function2
    def vis_samples(self, drug_id, tissue_id, tar_ccls, rnd_seed, k):
        if k > 0:
            # 模型fine-tune
            support_drug_fea, support_ccl_fea, support_y = self.get_support_samples(drug_id, tissue_id, k, tar_ccls, rnd_seed)
            f_hyper_net, support_loss_innerloop = self.inner_loop(task_x=(support_drug_fea.to(self.args.device), support_ccl_fea.to(self.args.device)), 
                                                                  task_y=(support_y).to(self.args.device), 
                                                                  model=self.meta_model)
        else:
            f_hyper_net = higher.patch.monkeypatch(
                            module=self.meta_model["hyper_net"],
                            copy_initial_weights=False,
                            track_higher_grads=True if self.args.mode=="train" else False)
        tar_drug_fea_ls, tar_ccl_fea_ls = self.get_target_samples(drug_id, tar_ccls)
        print("Start to explain...")
        for i, tar_ccl in enumerate(tar_ccls):
            save_path = osp.join(self.args.saved_path, "%s_%s" % (drug_id, tissue_id), "%s_%s" % (drug_id, tar_ccl), "%s" % self.args.task_name)
            if not osp.exists(save_path):
                os.makedirs(save_path)
            
            model_hooks_tmp = {
                "cd_embed":None,
                "d_embed":None,
                }
            
            model_add_hooks2(model=self.meta_model["f_base_net"], model_hooks=model_hooks_tmp)
            tar_drug_fea = collator_forDrug([tar_drug_fea_ls[i]]).to(self.args.device)
            tar_ccl_fea = tar_ccl_fea_ls[i].unsqueeze(dim=0).to(self.args.device)
            
            _ = self.prediction(x=(tar_drug_fea, tar_ccl_fea), adapted_hyper_net=f_hyper_net, model=self.meta_model)

            np.save(osp.join(save_path, "cd_embed_k%d.npy" % k), model_hooks_tmp["cd_embed"].squeeze().numpy())
            print(model_hooks_tmp["cd_embed"].squeeze().numpy().shape)
            
            np.save(osp.join(save_path, "d_embed_k%d.npy" % k), model_hooks_tmp["d_embed"].squeeze().numpy())
            print(model_hooks_tmp["d_embed"].squeeze().numpy().shape)
        print("Finished.")        
        return None    
    
    def get_src_support_samples(self, drug_id, tissue_id, k, rnd_seed):
        # 输入药物ID，组织ID，以及要获得的support samples的数量k
        # 输出support samples
        drug_dict = np.load(osp.join(self.src_drug_data_dir, "drug_dict.npy"), allow_pickle=True).item()
        ccl_dict = np.load(osp.join(self.src_ccl_data_path), allow_pickle=True).item()
        drug_path = osp.join(self.src_dr_data_dir, drug_id, "%s-%s.csv"%(drug_id, tissue_id))
        drug_subset = pd.read_csv(drug_path, header=0)
        
        rng = np.random.RandomState(rnd_seed)
        sample_idx = rng.choice([i for i in range(len(drug_subset))], k)
        support_df = drug_subset.iloc[sample_idx, :]
        
        drug_fea = drug_dict[drug_id]
        support_drug_fea = collator_forDrug([drug_fea for i in range(k)])
        support_ccl_fea = [ccl_dict[ccl_item] for ccl_item in support_df["DepMap_ID"]]
        support_ccl_fea = torch.stack(support_ccl_fea, dim=0)
        support_y = torch.tensor(support_df["IC50"].values.tolist())
        return support_drug_fea, support_ccl_fea, support_y
    
    def crox_predict(self, drug_id, tar_ccls, k, rnd_seed, src_drug_id, src_tissue_id):
        
        # 获取support ccls
        drug_path = osp.join(self.src_dr_data_dir, src_drug_id, "%s-%s.csv"%(src_drug_id, src_tissue_id))
        if not osp.exists(drug_path):
            return None
        else:
            drug_subset = pd.read_csv(drug_path, header=0)
            if len(drug_subset) < self.args.k:
                return None
            else:
                support_drug_fea, support_ccl_fea, support_y = self.get_src_support_samples(src_drug_id, src_tissue_id, k, rnd_seed)
                f_hyper_net, _ = self.inner_loop(task_x=(support_drug_fea.to(self.args.device), support_ccl_fea.to(self.args.device)), 
                                                                      task_y=(support_y).to(self.args.device), 
                                                                      model=self.meta_model)
                
                tar_drug_fea_ls, tar_ccl_fea_ls = self.get_target_samples(drug_id, tar_ccls)
                tar_drug_feas = collator_forDrug(tar_drug_fea_ls).to(self.args.device)
                tar_ccl_feas = torch.stack(tar_ccl_fea_ls, dim=0).to(self.args.device)
                logits = self.prediction(x=(tar_drug_feas, tar_ccl_feas), adapted_hyper_net=f_hyper_net, model=self.meta_model)
                return logits
    
    def crox_explain_samples(self, drug_id, tar_ccls, k, rnd_seed, src_drug_id, src_tissue_id):
        
        # 获取support ccls
        drug_path = osp.join(self.src_dr_data_dir, src_drug_id, "%s-%s.csv"%(src_drug_id, src_tissue_id))
        if not osp.exists(drug_path):
            return None
        else:
            drug_subset = pd.read_csv(drug_path, header=0)
            if len(drug_subset) < self.args.k:
                return None
            else:
                support_drug_fea, support_ccl_fea, support_y = self.get_src_support_samples(src_drug_id, src_tissue_id, k, rnd_seed)
                f_hyper_net, _ = self.inner_loop(task_x=(support_drug_fea.to(self.args.device), support_ccl_fea.to(self.args.device)), 
                                                                      task_y=(support_y).to(self.args.device), 
                                                                      model=self.meta_model)
                
                tar_drug_fea_ls, tar_ccl_fea_ls = self.get_target_samples(drug_id, tar_ccls)

                for i, tar_ccl in enumerate(tar_ccls):
                    save_path = osp.join(self.args.saved_path, "%s_%s" % (drug_id, tar_ccl), "%s_%s_rm%d" % (drug_id, tar_ccl, rnd_seed))
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    
                    model_hooks_tmp = {
                        "drug_attn":[],
                        "pathway_attn":[],
                        "cd_attn":[],
                        "ccl_weights":None,
                        }
                    
                    model_add_hooks1(model=self.meta_model["f_base_net"], model_hooks=model_hooks_tmp)
                    tar_drug_fea = collator_forDrug([tar_drug_fea_ls[i]]).to(self.args.device)
                    tar_ccl_fea = tar_ccl_fea_ls[i].unsqueeze(dim=0).to(self.args.device)
                    
                    _ = self.prediction(x=(tar_drug_fea, tar_ccl_fea), adapted_hyper_net=f_hyper_net, model=self.meta_model)

                    tmp_mat = np.stack(model_hooks_tmp["drug_attn"], axis=0)
                    np.save(osp.join(save_path, "drug_attn.npy"), tmp_mat.squeeze())
                    print(tmp_mat.squeeze().shape)
                    
                    tmp_mat = np.stack(model_hooks_tmp["pathway_attn"], axis=0)
                    np.save(osp.join(save_path, "pathway_attn.npy"), tmp_mat.squeeze())
                    print(tmp_mat.squeeze().shape)
                    
                    tmp_mat = np.stack(model_hooks_tmp["cd_attn"], axis=0)
                    np.save(osp.join(save_path, "cd_attn.npy"), tmp_mat.squeeze())
                    print(tmp_mat.squeeze().shape)
                    
                    np.save(osp.join(save_path, "ccl_weights.npy"), model_hooks_tmp["ccl_weights"].numpy())
                    print(model_hooks_tmp["ccl_weights"].numpy().shape)
                print("Finished.")
                return None
    
    def crox_viz_samples(self, drug_id, tar_ccls, k, rnd_seed, src_drug_id, src_tissue_id):
        # 获取support ccls
        drug_path = osp.join(self.src_dr_data_dir, src_drug_id, "%s-%s.csv"%(src_drug_id, src_tissue_id))
        if not osp.exists(drug_path):
            return None
        else:
            drug_subset = pd.read_csv(drug_path, header=0)
            if len(drug_subset) < self.args.k:
                return None
            else:
                support_drug_fea, support_ccl_fea, support_y = self.get_src_support_samples(src_drug_id, src_tissue_id, k, rnd_seed)
                f_hyper_net, _ = self.inner_loop(task_x=(support_drug_fea.to(self.args.device), support_ccl_fea.to(self.args.device)), 
                                                                      task_y=(support_y).to(self.args.device), 
                                                                      model=self.meta_model)
                
                tar_drug_fea_ls, tar_ccl_fea_ls = self.get_target_samples(drug_id, tar_ccls)
                print("Start to explain...")
                for i, tar_ccl in enumerate(tar_ccls):
                    save_path = osp.join(self.args.saved_path, "%s_%s" % (drug_id, src_tissue_id), "%s_%s" % (drug_id, tar_ccl), "%s" % self.args.task_name)
                    if not osp.exists(save_path):
                        os.makedirs(save_path)
                    
                    model_hooks_tmp = {
                        "cd_embed":None,
                        "d_embed":None,
                        }
                    
                    model_add_hooks2(model=self.meta_model["f_base_net"], model_hooks=model_hooks_tmp)
                    tar_drug_fea = collator_forDrug([tar_drug_fea_ls[i]]).to(self.args.device)
                    tar_ccl_fea = tar_ccl_fea_ls[i].unsqueeze(dim=0).to(self.args.device)
                    
                    _ = self.prediction(x=(tar_drug_fea, tar_ccl_fea), adapted_hyper_net=f_hyper_net, model=self.meta_model)

                    np.save(osp.join(save_path, "cd_embed_k%d.npy" % k), model_hooks_tmp["cd_embed"].squeeze().numpy())
                    print(model_hooks_tmp["cd_embed"].squeeze().numpy().shape)
                    
                    np.save(osp.join(save_path, "d_embed_k%d.npy" % k), model_hooks_tmp["d_embed"].squeeze().numpy())
                    print(model_hooks_tmp["d_embed"].squeeze().numpy().shape)
                print("Finished.")        
                return None    
                
            
    @staticmethod
    def torch_module_to_functional(torch_net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
        """Convert a conventional torch module to its "functional" form
        """
        f_net = higher.patch.make_functional(module=torch_net)
        higher.patch.buffer_sync(module=torch_net, fmodule=f_net)
        f_net.track_higher_grads = False
        f_net._fast_params = [[]]
        return f_net
    
    
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