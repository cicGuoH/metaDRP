# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:56:12 2023

@author: GUI
"""


import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from models.graphormer import GraphormerEncoder
from models.ccl_encoder import CCLEncoder
from models.layers import MultiHeadAttention, FFN
from torch_geometric.utils import get_laplacian

def get_connectivity(num_pathways):
    connectivity = torch.load("../data_processed/MSigDB_p%s/connectivity.pt" % num_pathways)
    return connectivity

def get_ccl_adj(edge_index):
    i, v = get_laplacian(edge_index.long(), normalization="sym")
    num_nodes = int(edge_index.max().numpy()) + 1
    adj_sym = torch.sparse_coo_tensor(i, v, torch.Size([num_nodes, num_nodes])).to_dense()
    return adj_sym

class GraphLevelGraphormer(nn.Module):
    def __init__(self, args, output_dim=1):
        super().__init__()
        self.args = args
        self.graph_encoder = GraphormerEncoder(num_atoms=args.drug_num_atoms, 
                                          num_in_degree=args.drug_num_in_degree, 
                                          num_out_degree=args.drug_num_out_degree, 
                                          num_edges=args.drug_num_edges, 
                                          num_spatial=args.drug_num_spatial, 
                                          num_edge_dist=args.drug_num_edge_dist, 
                                          multi_hop_max_dist=args.drug_multi_hop_max_dist,
                                          n_encode_layers = args.drug_n_encode_layers,
                                          embed_dim = args.drug_embed_dim, 
                                          ffn_embed_dim = args.embed_dim, 
                                          num_heads = args.drug_num_heads,
                                          dropout_rate = args.dropout_rate,
                                          attention_dropout_rate = args.dropout_rate,
                                          activation_dropout_rate = args.dropout_rate,
                                          layer_dropout_rate = args.layer_dropout_rate,
                                          encoder_normalize_before = False,
                                          pre_layernorm = False,
                                          apply_graphormer_init = False,
                                          activation_fn = "gelu",
                                          embed_scale = None,
                                          freeze_embeddings = False,
                                          n_trans_layers_to_freeze = 0,
                                          traceable = False,
                                          q_noise = args.q_noise,
                                          qn_block_size = args.qn_block_size,
                                          )
        
        self.fc = nn.Sequential(
            nn.ELU(inplace=True),
            nn.Linear(args.embed_dim, output_dim),
            )
        
    def forward(self, inputx):
        x = self.graph_encoder(inputx)[1]
        x = self.fc(x)
        return x

class CCLPredictor(nn.Module):
    def __init__(self, args, output_dim, pathway_name="KEGG"):
        super().__init__()
        self.args = args
        self.ccl_enc = CCLEncoder(num_genes=args.num_genes, 
                                  in_features=args.in_features, 
                                  pos_max=args.pos_max, 
                                  pos_embed_dim=args.pos_embed_dim, 
                                  embed_dim=args.embed_dim, 
                                  dropout_rate=args.dropout_rate, 
                                  num_heads=args.num_heads, 
                                  num_pattn_layers=args.num_pattn_layers, 
                                  pre_ln=False, 
                                  activation_fn="gelu")
        self.conv = nn.Sequential(
                    nn.Dropout(args.dropout_rate),
                    nn.Conv1d(in_channels=args.embed_dim, out_channels=1, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Dropout(args.dropout_rate),
                    )
        
        self.out = nn.Linear(args.ccl_num_pathways, output_dim)
    def forward(self, inputx):
        ccl_input, ccl_adj = inputx
        x = self.ccl_enc(ccl_input, ccl_adj)[0]
        x = torch.squeeze(self.conv(x.transpose(1,2)), dim=1)
        x = self.out(x)
        return x
    
class metaDRP(nn.Module):
    def __init__(self, args, ccl_drug_trans=True):
        super().__init__()
        
        self.args = args
        self.ccl_drug_trans = ccl_drug_trans
        connectivity = get_connectivity(args.num_pathways).to(self.args.device)
        edge_index = torch.load("../data_processed/MSigDB_p%d/edge_index.pt" % args.num_pathways)
        adj_sym = get_ccl_adj(edge_index).to(self.args.device)
        self.drug_enc = GraphormerEncoder(num_atoms=args.drug_num_atoms, 
                                          num_in_degree=args.drug_num_in_degree, 
                                          num_out_degree=args.drug_num_out_degree, 
                                          num_edges=args.drug_num_edges, 
                                          num_spatial=args.drug_num_spatial, 
                                          num_edge_dist=args.drug_num_edge_dist, 
                                          multi_hop_max_dist=args.drug_multi_hop_max_dist,
                                          n_encode_layers = args.drug_n_encode_layers,
                                          embed_dim = args.drug_embed_dim, 
                                          ffn_embed_dim = args.embed_dim, 
                                          num_heads = args.drug_num_heads,
                                          dropout_rate = args.dropout_rate,
                                          attention_dropout_rate = args.dropout_rate,
                                          activation_dropout_rate = args.dropout_rate,
                                          layer_dropout_rate = args.layer_dropout_rate,
                                          encoder_normalize_before = False,
                                          pre_layernorm = False,
                                          apply_graphormer_init = False,
                                          activation_fn = "gelu",
                                          embed_scale = None,
                                          freeze_embeddings = False,
                                          n_trans_layers_to_freeze = 0,
                                          traceable = False,
                                          q_noise = args.q_noise,
                                          qn_block_size = args.qn_block_size,
                                          )
        
        self.ccl_enc =CCLEncoder(num_genes=args.num_genes, 
                                 in_features=args.in_features, 
                                 embed_dim=args.embed_dim, 
                                 num_pathways=args.num_pathways,
                                 dropout_rate=args.dropout_rate, 
                                 num_heads=args.ccl_num_heads, 
                                 num_pattn_layers=args.num_pattn_layers, 
                                 adj_sym = adj_sym,
                                 pre_ln=False, 
                                 connectivity=connectivity,
                                 activation_fn="gelu")
        if self.ccl_drug_trans:
            self.CDTrans = MultiHeadAttention(embed_dim=args.embed_dim, 
                                              num_heads=args.cd_num_heads,
                                              self_attention=True, 
                                              q_noise=args.q_noise,
                                              qn_block_size=args.qn_block_size,
                                              )
            self.CDTransDrop = nn.Dropout(args.dropout_rate)
        
        self.ffn = FFN(in_features=args.embed_dim, hidden_features=args.embed_dim//2, out_features=1)
        self.ln_ffn = nn.LayerNorm(args.num_pathways+1)
        self.op = FFN(in_features=args.num_pathways+1, hidden_features=64, out_features=1)
        
    def forward(self, inputx):
        drug_fea, ccl_fea = inputx
        x_ccl = self.ccl_enc(ccl_fea)  # BxPxE
        x_drug = self.drug_enc(drug_fea)[1] # BxE
        # concat
        x = torch.concat((x_ccl, torch.unsqueeze(x_drug, 1)), dim=1) # Bx(P+1)xE
        if self.ccl_drug_trans:
            residual = x
            x = x.transpose(0,1)
            x, cd_attn = self.CDTrans(query=x,
                             attn_bias=None,
                             need_weights=True)
            x = x.transpose(0,1)
            x = self.CDTransDrop(residual+x) # Bx(P+1)xE
        
        x = torch.squeeze(self.ffn(x), -1) # Bx(P+1)
        x = self.ln_ffn(x)
        x = self.op(x).squeeze(dim=-1)
        return x
    
        
    