# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:00:23 2023

@author: GUI
"""
from typing import Callable, Optional, Tuple
import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from models.layers import GraphNodeFeature, GraphAttnBias, MultiHeadAttention, quant_noise, LayerDropModuleList, GraphormerGraphEncoderLayer

def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiHeadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
        

class GraphormerEncoder(nn.Module):
    def __init__(self,
                 num_atoms: int,
                 num_in_degree: int,
                 num_out_degree: int,
                 num_edges: int,
                 num_spatial: int, 
                 num_edge_dist: int,
                 multi_hop_max_dist: int,
                 edge_type: str="multi_hop",
                 n_encode_layers: int=12,
                 embed_dim: int=768,
                 ffn_embed_dim: int = 768,
                 num_heads: int = 32,
                 dropout_rate: float=0.1,
                 attention_dropout_rate: float=0.1,
                 activation_dropout_rate: float = 0.1,
                 layer_dropout_rate: float = 0.0,
                 encoder_normalize_before: bool = False,
                 pre_layernorm: bool = False,
                 apply_graphormer_init: bool = True,
                 activation_fn: str = "gelu",
                 embed_scale: float = None,
                 freeze_embeddings: bool = False,
                 n_trans_layers_to_freeze: int = 0,
                 traceable: bool = False,
                 q_noise: float = 0.0,
                 qn_block_size: int = 8,
                 ):
        super().__init__()
        
        
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim
        self.layer_dropout_rate = layer_dropout_rate
        self.apply_graphormer_init = apply_graphormer_init
        self.dropout_module = nn.Dropout(dropout_rate)
        self.traceable = traceable
        self.graph_node_feature = GraphNodeFeature(num_atoms=num_atoms, 
                                                   num_in_degree=num_in_degree, 
                                                   num_out_degree=num_out_degree, 
                                                   hidden_dim=embed_dim, 
                                                   n_layers=n_encode_layers)
        self.graph_attn_bias = GraphAttnBias(num_heads=num_heads, 
                                             num_edges=num_edges, 
                                             num_spatial=num_spatial, 
                                             num_edge_dist=num_edge_dist, 
                                             edge_type=edge_type, 
                                             multi_hop_max_dist=multi_hop_max_dist, 
                                             n_layers=n_encode_layers)
        
        if q_noise > 0:
            self.quant_noise = quant_noise(
                module=nn.Linear(self.embed_dim, self.embed_dim), 
                p=q_noise, 
                block_size=qn_block_size)
        else:
            self.quant_noise = None
        
        if encoder_normalize_before:
            self.embed_layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.embed_layer_norm = None
            
        if pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        
        if self.layer_dropout_rate > 0:
            self.layers = LayerDropModuleList(p = self.layer_dropout_rate)
        else:
            self.layers = nn.ModuleList([])
            
        self.layers.extend(
            [
                self.build_graphormer_encoder_layer(embed_dim=embed_dim, 
                                                    ffn_embed_dim=ffn_embed_dim, 
                                                    num_heads=num_heads, 
                                                    dropout_rate=dropout_rate, 
                                                    attention_dropout=attention_dropout_rate, 
                                                    activation_dropout_rate=activation_dropout_rate, 
                                                    activation_fn=activation_fn, 
                                                    q_noise=q_noise, 
                                                    qn_block_size=qn_block_size, 
                                                    pre_layernorm=pre_layernorm)
                for _ in range(n_encode_layers)
                ]
            )
        
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)    
        
        
        def freeze_module_params(m):
                if m is not None:
                    for p in m.parameters():
                        p.requires_grad = False
        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])
            
    
    def forward(self, 
                batched_data,
                perturb=None,
                last_state_only:bool=False,
                token_embeddings:Optional[Tensor]=None,
                attn_mask:Optional[Tensor]=None,
                ) -> Tuple[Tensor, Tensor]:
        
        # 计算padding mask
        data_x = batched_data.x
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        
        if token_embeddings is not None:
            x = token_embeddings
        
        else:
            x = self.graph_node_feature(batched_data)
            
        if perturb is not None:
            x[:, 1:, :] += perturb
            
        attn_bias = self.graph_attn_bias(batched_data)
        if self.embed_scale is not None:
            x = x * self.embed_scale
        
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        
        if self.embed_layer_norm is not None:
            x = self.embed_layer_norm(x)
        
        x = self.dropout_module(x)
        
        x = x.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append((x, attn))
        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep
        

    def build_graphormer_encoder_layer(self,
                                       embed_dim,
                                       ffn_embed_dim,
                                       num_heads,
                                       dropout_rate,
                                       attention_dropout,
                                       activation_dropout_rate,
                                       activation_fn,
                                       q_noise,
                                       qn_block_size,
                                       pre_layernorm,
                                       ):
        return GraphormerGraphEncoderLayer(embed_dim=embed_dim,
                                           ffn_embed_dim=ffn_embed_dim,
                                           num_heads=num_heads,
                                           dropout_rate=dropout_rate,
                                           attention_dropout_rate=attention_dropout,
                                           activation_dropout_rate=activation_dropout_rate,
                                           activation_fn=activation_fn,
                                           q_noise=q_noise,
                                           qn_block_size=qn_block_size,
                                           pre_layernorm=pre_layernorm)
    
    
