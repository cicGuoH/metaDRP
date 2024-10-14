
# -*- coding: utf-8 -*-
"""
@article{shi2022benchmarking,
  title={Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets},
  author={Yu Shi and Shuxin Zheng and Guolin Ke and Yifei Shen and Jiacheng You and Jiyan He and Shengjie Luo and Chang Liu and Di He and Tie-Yan Liu},
  journal={arXiv preprint arXiv:2203.04810},
  year={2022},
  url={https://arxiv.org/abs/2203.04810}
}

@inproceedings{
ying2021do,
title={Do Transformers Really Perform Badly for Graph Representation?},
author={Chengxuan Ying and Tianle Cai and Shengjie Luo and Shuxin Zheng and Guolin Ke and Di He and Yanming Shen and Tie-Yan Liu},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=OeWooOxFwDa}
}

"""
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
import torch.utils.checkpoint as cp
from typing import Callable, Optional, Tuple

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )

def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)

def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)

def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class GraphNodeFeature(nn.Module):
    """
    Centrality Encoding
    xi = x + z_in + z_out
    
    用embedding初始化各图中各节点的feature
    """
    def __init__(self, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers):
        super().__init__()
        self.num_atoms = num_atoms
        
        # 1 for graph token(虚拟节点表示图整体的信息)
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.token_init = nn.Parameter(torch.zeros((1,1), dtype=torch.long), requires_grad=False) 
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        
    
    def forward(self, batched_data):
        
        x, in_degree, out_degree = batched_data.x, batched_data.in_degree, batched_data.out_degree
        
        n_graph, n_node = x.size()[:2] # [B, T, 9]
        
        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        
        node_feature = (
                node_feature
                + self.in_degree_encoder(in_degree)  # [n_graph, n_node, n_hidden]
                + self.out_degree_encoder(out_degree)  # [n_graph, n_node, n_hidden]
        )
        
        graph_token_feature = self.graph_token(self.token_init.long()).repeat(n_graph, 1, 1)
        # graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        
        return graph_node_feature

class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    
    计算途中个节点的feature，输出是[num_graph, num_heads, num_nodes, num_nodes]
    
    空间编码：
    1. spatial_pos记录了两个节点之间最短路径长度。
    2. 使用Embedding模块（第一个参数是词嵌入字典大小， 第二个参数是每个词嵌入向量的大小）
    
    """

    def __init__(
        self,
        num_heads,
        num_edges,
        num_spatial,
        num_edge_dist,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dist * num_heads * num_heads, 1
            )
            self.edge_dis_init = nn.Parameter(torch.zeros((1,num_edge_dist * num_heads * num_heads), dtype=torch.long), requires_grad=False)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        self.graph_token_virtual_distance_init = nn.Parameter(torch.zeros((1,1), dtype=torch.long), requires_grad=False)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data.attn_bias,
            batched_data.spatial_pos,
            batched_data.x,
        )
        
        # avoid nan
        attn_bias = torch.zeros_like(attn_bias)
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = (
            batched_data.edge_input,
            batched_data.attn_edge_type,
        )

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance(self.graph_token_virtual_distance_init.long()).view(1, self.num_heads, 1)
        # t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            """
            multi-hop指定边的类型，用于确定图中两个节点之间多跳边的最大距离，越大模型计算量越大
            """
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder(self.edge_dis_init.long()).reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            
            # edge_input_flat = torch.bmm(
            #     edge_input_flat,
            #     self.edge_dis_encoder.weight.reshape(
            #         -1, self.num_heads, self.num_heads
            #     )[:max_dist, :, :],
            # )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
        return graph_attn_bias

  
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., drop_act=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.drop_act = nn.Dropout(drop_act)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Not used
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

# Not used
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    
    正则化方法
    将深度学习模型中的多分支结构随机“失效”（不同于Dropout对神经元随机“失效”）；
    在drop_path分支中，每个batch由drop_prob的概率样本在drop_path中不会执行，而是以0直接传递
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim,
                 num_heads,
                 qdim=None,
                 kdim=None, 
                 vdim=None,
                 dropout_rate=0.,
                 bias=True, 
                 self_attention=False,
                 q_noise=0.,
                 qn_block_size=8,):
        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        
        self.dropout_module = nn.Dropout(dropout_rate)
        self.head_dim = embed_dim // num_heads
        assert ( self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        
        self.self_attention = self_attention
        assert self.self_attention, "Only support self attention"
        assert not self.self_attention or self.qkv_same_dim, ( "Self-attention requires query, key and " "value to be of the same size")
        
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(self.qdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
            
    
    def forward(self, 
                query,
                attn_bias:Optional[Tensor],
                key_padding_mask:Optional[Tensor]=None,
                need_weights:bool=True, #是否返回平均的注意力矩阵
                attn_mask:Optional[Tensor]=None,
                before_softmax: bool=False, #是否返回没有经过softmax的注意力注意力矩阵
                need_head_weights: bool=True, #是否返回每个head的注意力矩阵
                ) -> Tuple[Tensor, Optional[Tensor]]: 
        
        if need_head_weights:
            need_weights = True
        
        
        tgt_len, bsz, _ = query.size()
        src_len = tgt_len

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling
        
        
        q = (q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        
        if k is not None:
            k = (k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        if v is not None:
            v = (v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))
            
        assert k is not None
        assert k.size(1) == src_len
        
        
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
            
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask   
        
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        if before_softmax:
            return attn_weights, v
        
        attn_weights_float = attn_weights.softmax(dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights
    
    
    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
            
            
# refer to https://github.com/facebookresearch/fairseq/blob/98ebe4f1ada75d006717d84f9d603519d8ff5579/fairseq/modules/layer_drop.py
class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.
    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m    
        
class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        activation_dropout_rate: float = 0.1,
        activation_fn: str = "relu",
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        pre_layernorm: bool = False,
    ):
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_dropoutrate_rate = attention_dropout_rate
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.pre_layernorm = pre_layernorm

        self.dropout_module = nn.Dropout(dropout_rate)
        self.activation_dropout_module = nn.Dropout(activation_dropout_rate)

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiHeadAttention(self.embed_dim,
                                            num_heads,
                                            dropout_rate=attention_dropout_rate,
                                            self_attention=True,
                                            q_noise=q_noise,
                                            qn_block_size=qn_block_size,
                                            )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = quant_noise(nn.Linear(self.embed_dim, ffn_embed_dim), p=q_noise, block_size=qn_block_size)
        self.fc2 = quant_noise(nn.Linear(ffn_embed_dim, self.embed_dim), p=q_noise, block_size=qn_block_size)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn        

