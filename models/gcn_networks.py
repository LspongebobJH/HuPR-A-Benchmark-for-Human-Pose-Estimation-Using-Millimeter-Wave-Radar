import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric_temporal import TGCN
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv, ChebConv, GATConv

class GCN_layers(nn.Module):
    def __init__(self, in_features, out_features, numKeypoints, bias=True):
        super(GCN_layers, self).__init__()
        self.bias = bias
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features, numKeypoints))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, adj)
        output = torch.matmul(self.weight, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class PRGCN(nn.Module):
    def __init__(self, cfg, A):
        super(PRGCN, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.featureSize = (self.height//2) * (self.width//2)
        self.L1 = GCN_layers(self.featureSize, self.featureSize, self.numKeypoints)
        self.L2 = GCN_layers(self.featureSize, self.featureSize, self.numKeypoints)
        self.L3 = GCN_layers(self.featureSize,self.featureSize, self.numKeypoints)
        self.A = A
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def generate_node_feature(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = x.reshape(-1, self.numKeypoints, self.featureSize).permute(0, 2, 1)
        return x

    def gcn_forward(self, x):
        #x: (B, numFilters, numkeypoints)
        x2 = self.relu(self.L1(x, self.A))
        x3 = self.relu(self.L2(x2, self.A))
        keypoints = self.L3(x3, self.A)
        return keypoints.permute(0, 2, 1)

    def forward(self, x, *args):
        nodeFeat = self.generate_node_feature(x) # node features have been sequeezed into 1d vector
        heatmap = self.gcn_forward(nodeFeat).reshape(-1, self.numKeypoints, (self.height//2), (self.width//2))
        heatmap = F.interpolate(heatmap, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.sigmoid(heatmap).unsqueeze(1)

class TempPRGCN(nn.Module):
    def __init__(self, cfg, A):
        super(TempPRGCN, self).__init__()
        self.numGroupFrames = cfg.DATASET.numGroupFrames
        self.numFilters = cfg.MODEL.numFilters
        self.width = cfg.DATASET.heatmapSize
        self.height = cfg.DATASET.heatmapSize
        self.numKeypoints = cfg.DATASET.numKeypoints
        self.featureSize = (self.height//2) * (self.width//2)
        self.A, _ = dense_to_sparse(A)
        self.n_layers = cfg.MODEL.n_layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            # self.layers.append(GCN_layers(self.featureSize, self.featureSize, self.numKeypoints))
            self.layers.append(GCNConv(self.featureSize, self.featureSize))

        self.temp = TGCN(self.featureSize, self.featureSize)
        self.temp_back = TGCN(self.featureSize, self.featureSize)

    def generate_node_feature(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = x.reshape(self.numGroupFrames, self.numKeypoints, self.featureSize)
        return x

    def forward(self, feat, video_id):
        feat = self.generate_node_feature(feat)
        for layer in self.layers:
            feat = self.relu(layer(feat, self.A))
        
        prev_idx = -1
        H_list = []
        for frame_feat, idx in zip(feat, video_id):
            if prev_idx != idx: 
                # the current idx and the previous idx are not the same, 
                # or there's no previous idx
                # then initialize the hidden state to None 
                H = None
            prev_idx = idx
            H = self.temp(frame_feat, self.A, H=H)
            H_list.append(H)
        H_list = torch.stack(H_list, dim=0)

        post_idx = -1
        H_back_list  = []
        for frame_feat, idx in zip(torch.flip(feat, dims=(0,)), video_id[::-1]):
            if post_idx != idx:
                H_back = None
            post_idx = idx
            H_back = self.temp_back(frame_feat, self.A, H=H_back)
            H_back_list.append(H_back)
        H_back_list.reverse()
        H_back_list = torch.stack(H_back_list, dim=0)

        H_list = H_list + H_back_list
        H_list = H_list.reshape(-1, self.numKeypoints, (self.height//2), (self.width//2))
        H_list = F.interpolate(H_list, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.sigmoid(H_list).unsqueeze(1)
        
        
    

