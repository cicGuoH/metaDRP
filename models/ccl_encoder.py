# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from models.layers import MultiHeadAttention, get_activation_fn, FFN
import scipy.sparse as sp
import numpy as np
import pandas as pd
import os.path as osp
import torch_sparse
    
class SparseLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, connectivity=None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        nnz = connectivity.shape[1]
        coalesce_device = connectivity.device 
        self.sparsity = nnz/(out_features*in_features)
        values = torch.empty(nnz, device=coalesce_device)
        indices, values = torch_sparse.coalesce(connectivity, values, out_features, in_features)
        self.indices = indices
        # self.register_buffer('indices', indices.cpu())
        self.weights = nn.Parameter(values.cpu())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        bound = 1 / self.in_features**0.5
        nn.init.uniform_(self.weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        
    @property
    def weight(self):
        """ returns a torch.sparse.FloatTensor view of the underlying weight matrix
            This is only for inspection purposes and should not be modified or used in any autograd operations
        """
        weight = torch.sparse.FloatTensor(self.indices, self.weights, (self.out_features, self.in_features))
        return weight.coalesce().detach()
    
    def forward(self, inputs):
        output_shape = list(inputs.shape)
        output_shape[-1] = self.out_features
        inputs = inputs.flatten(end_dim=-2)
        sparseMatrix = torch.sparse.FloatTensor(self.indices,
                                                self.weights,
                                                torch.Size([self.out_features, self.in_features]))
        output = torch.sparse.mm(sparseMatrix, inputs.t()).t()
        if self.bias is not None:
            output += self.bias
        return output.view(output_shape)
    
class GraphConvolutionLayer(nn.Module):
    """
    simple GCN: h = A_sym*x*W, Specifically, it is modified to satisfy static graph in a batch way.
    """

    def __init__(self, in_features, out_features, bias=True, pre_ln=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pre_ln = pre_ln
        
        if self.pre_ln:
            self.pre_layernorm = nn.LayerNorm(in_features)
            
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputx, adj):
        if self.pre_ln:
            x = self.pre_layernorm(inputx)
        else:
            x = inputx
            
        if inputx.dim() == 3: # B x N x C
            support = torch.matmul(x, self.weight).transpose(1,2)
            output = torch.matmul(support, adj).transpose(1,2)
        else:
            assert inputx.dim() == 2, "input is a graph or pyG batched graph so that shape of inputx is (B*N)xC"
            support = torch.mm(x, self.weight) # N x E
            output = torch.mm(adj, support) # N x E
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'     
               
               
class PAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, pre_ln, activation_fn):
        self.pre_ln = pre_ln
        super().__init__()
        self.mhAttn = MultiHeadAttention(embed_dim=embed_dim, 
                                         num_heads=num_heads,
                                         dropout_rate=dropout_rate,
                                         self_attention=True,
                                         )
        self.dropout_module = nn.Dropout(dropout_rate)
        self.activation_fn = get_activation_fn(activation_fn)
        self.ln = nn.LayerNorm(embed_dim)
        self.fi_ln = nn.LayerNorm(embed_dim)
        self.ffn = FFN(in_features=embed_dim, 
                       hidden_features=embed_dim, 
                       out_features=embed_dim,
                       drop=dropout_rate,
                       drop_act=dropout_rate)
    def forward(self, x):
        residual = x
        if self.pre_ln:
            x = self.ln(x)
        x, attn = self.mhAttn(query=x,
                              attn_mask=None, 
                              attn_bias=None, 
                              need_weights=True)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_ln:
            x = self.ln(x)
        residual = x
        if self.pre_ln:
            x = self.fi_ln(x)
        x = self.ffn(x)
        x = residual + x
        if not self.pre_ln:
            x = self.fi_ln(x)
        return x, attn

    
class CCLEncoder(nn.Module):
    def __init__(self, 
                 num_genes, 
                 in_features, 
                 embed_dim,
                 dropout_rate, 
                 num_heads, 
                 num_pathways, 
                 num_pattn_layers, 
                 pre_ln, 
                 connectivity,
                 adj_sym, 
                 max_hop=3,
                 activation_fn="gelu"):
        
        self.in_features = in_features # num_omics
        self.embed_dim = embed_dim  # gene_pos_embed_dim
        self.max_hop = 3
        self.adj_sym = adj_sym
        super().__init__()
    
        gcn_init = GraphConvolutionLayer(in_features=self.in_features, out_features=self.embed_dim, pre_ln=pre_ln)
        self.gcn_layers = nn.ModuleList([gcn_init])
        if self.max_hop > 1:
            self.gcn_layers.extend([
                GraphConvolutionLayer(in_features=self.embed_dim, out_features=self.embed_dim, pre_ln=pre_ln) for _ in range(max_hop-1)
                ])
        self.splinear = SparseLinear(num_genes, num_pathways, connectivity=connectivity)
        self.ln_sp = nn.LayerNorm(num_pathways)
        self.activation_fn = get_activation_fn(activation_fn)
        self.activation_dropout_module = nn.Dropout(dropout_rate)
        self.pattn_layers = nn.ModuleList(
             self.build_pattn_layer(embed_dim=embed_dim, 
                                    num_heads=num_heads, 
                                    dropout_rate=dropout_rate, 
                                    pre_ln=pre_ln, 
                                    activation_fn=activation_fn) for _ in range(num_pattn_layers))
        
        self.ffn = FFN(in_features=embed_dim, 
                         hidden_features=embed_dim*2, 
                         out_features=embed_dim,
                         drop=dropout_rate,
                         drop_act=dropout_rate)
        
    def forward(self, ccl_fea):
        x = ccl_fea
        for i,layer in enumerate(self.gcn_layers):
            if i == 0:
                x = layer(x, self.adj_sym)
                residual = x
            else:
                x = layer(x, self.adj_sym)
                x = residual + x
                residual = x
        
        x = self.splinear(x.transpose(1, 2)) # n_ccls, embed_dim, n_pathway
        x = self.ln_sp(x).transpose(1, 2) # n_ccls, n_pathway, embed_dim
        x = self.activation_dropout_module(self.activation_fn(x))
        x = torch.transpose(x, 0, 1) # n_pathway, n_ccls, embed_dim
        for layer in self.pattn_layers:
            x, _ = layer(x)
        x = x.transpose(0, 1) # n_ccls, n_pathway, embed_dim
        x = self.ffn(x)
        return x
        
    
    def build_pattn_layer(self, embed_dim, num_heads, dropout_rate, pre_ln, activation_fn):
        return PAttn(embed_dim=embed_dim, 
                     num_heads=num_heads, 
                     dropout_rate=dropout_rate, 
                     pre_ln=pre_ln, 
                     activation_fn=activation_fn)
    
