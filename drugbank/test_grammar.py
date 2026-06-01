import torch
import torch.nn as nn

# pair_conv = torch.randn(512,1,128)
# pair_graph = torch.randn(512,1,128)
#
# mix_attention_layer = nn.MultiheadAttention(128, 8)
#
# pair_conv_QKV = pair_conv.permute(2, 0, 1)
# pair_graph_QKV = pair_graph.permute(2, 0, 1)
#
# heads_att, _ = mix_attention_layer(pair_conv, pair_graph, pair_graph)
# tails_att, _ = mix_attention_layer(pair_graph, pair_conv, pair_conv)
#
# # heads_att = heads_att.permute(1, 2, 0)
# # tails_att = tails_att.permute(1, 2, 0)
# print(heads_att.shape)
#
# headsConv = pair_conv * 0.5 + heads_att * 0.5
# tailsConv = pair_graph * 0.5 + tails_att * 0.5
#
# print(headsConv.shape)
# print(tailsConv.shape)


pair_conv = torch.randn(512,128,1)
pair_graph = torch.randn(512,128,1)
mix_attention_layer = nn.MultiheadAttention(128, 8)

conv_QKV = pair_conv.permute(2, 0, 1)
graph_QKV = pair_graph.permute(2, 0, 1)

conv_att, _ = mix_attention_layer(conv_QKV, graph_QKV, graph_QKV)
graph_att, _ = mix_attention_layer(graph_QKV, conv_QKV, conv_QKV)

conv_att = conv_att.permute(1, 2, 0)
graph_att = graph_att.permute(1, 2, 0)

pair_conv = pair_conv * 0.5 + conv_att * 0.5
pair_conv = pair_conv.squeeze(2)

pair_graph = pair_graph * 0.5 + graph_att * 0.5
pair_graph = pair_graph.squeeze(2)
print(pair_conv.shape,pair_graph.shape)