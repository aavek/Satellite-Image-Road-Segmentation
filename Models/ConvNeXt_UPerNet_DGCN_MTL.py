import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import Permute
import torchinfo
import cv2

ConvNeXt_Archs = {
                'Tiny': [[3, 3, 9, 3], [96, 192, 384, 768], 0.4, 1.0],       # [Depths, FeatureDimensions, DropPathRate, layerScale]
                'Small': [[3, 3, 27, 3], [96, 192, 384, 768], 0.4, 1.0],
                'Base': [[3, 3, 27, 3], [128, 256, 512, 1024], 0.4, 1.0],
                'Large': [[3, 3, 27, 3], [192, 384, 768, 1536], 0.4, 1.0],
                'XLarge': [[3, 3, 27, 3], [256, 512, 1024, 2048], 0.4, 1.0]
                }

class LayerNorm(nn.Module):
    def __init__(self, _shape, eps=1e-6, _format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(_shape))
        self.bias = nn.Parameter(torch.zeros(_shape))
        self.eps = eps
        self._format = _format
        self._shape = (_shape, )
    def forward(self, x):
        if self._format == "channels_last":
            return F.layer_norm(x, self._shape, self.weight, self.bias, self.eps)
        elif self._format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class PatchifyStem(nn.Sequential):
    def __init__(self, in_Channels, out_Channels):
        super().__init__(nn.Conv2d(in_Channels, out_Channels, kernel_size=4, stride=4),
                         LayerNorm(out_Channels, eps=1e-6, _format="channels_first"))

class DownsamplingConv(nn.Sequential):
    def __init__(self, FeatureDimensions, i):
        super().__init__(LayerNorm(FeatureDimensions[i], eps=1e-6, _format="channels_first"),
                         nn.Conv2d(FeatureDimensions[i], FeatureDimensions[i+1], kernel_size=2, stride=2))

class DropPath(nn.Module):
    def __init__(self, DPR, Training):
        super().__init__()
        self.DPR = DPR
        self.Training = Training
    def forward(self, x):
        if self.DPR == 0. or not self.Training:
            return x
        KeepProb = 1 - self.DPR
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(KeepProb)
        if KeepProb > 0.0:
            random_tensor.div(KeepProb)
        return x * random_tensor

class ConvNeXtBlock(nn.Module):
    def __init__(self, FDim, Expansion, DropPathRate, LayerScaleInitial, Training):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((FDim)) * LayerScaleInitial, requires_grad=True) if LayerScaleInitial > 0 else None
        self.dropPath = DropPath(DropPathRate, Training) if DropPathRate > 0.0 else nn.Identity()
        self.convNextBlock = nn.Sequential(nn.Conv2d(FDim, FDim, kernel_size = 7, padding = 3, groups = FDim),
                                           Permute([0, 2, 3, 1]),
                                           LayerNorm(FDim, eps=1e-6),
                                           nn.Linear(FDim, Expansion * FDim),
                                           nn.GELU(),
                                           nn.Linear(Expansion * FDim, FDim))
    def forward(self, x):
        Input = x
        x = self.convNextBlock(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        x = self.dropPath(x) + Input
        return x

class ConvNeXtStage(nn.Sequential):
    def __init__(self, Depth, FeatureDimension, Expansion, DropIndex, DropPathRates, LayerScaleInitial, Training):
        super().__init__(*[ConvNeXtBlock(FeatureDimension, 
                                         Expansion, 
                                         DropPathRates[DropIndex + j], 
                                         LayerScaleInitial, Training) 
                                         for j in range(Depth)])

class Encoder(nn.Module):
    """ Adapted from ConvNeXt
        A PyTorch implementation of : `A ConvNet for the 2020s`
          https://arxiv.org/pdf/2201.03545.pdf
          # From https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/backbone/convnext.py
    """
    def __init__(self, in_Channels, Depths, FeatureDimensions, StochasticDepthRate, LayerScaleInitial, Training):
        super().__init__()
        
        self.DownsampleLayers = nn.ModuleList()
        self.DownsampleLayers.append(PatchifyStem(in_Channels,FeatureDimensions[0]))
        for i in range(3): # 3 intermediate downsampling conv layers after stem as in original implementation
            self.DownsampleLayers.append(DownsamplingConv(FeatureDimensions, i))
        
        self.FeatureResolutionStages = nn.ModuleList()
        self.dropPathRates = [x.item() for x in torch.linspace(0, StochasticDepthRate, sum(Depths))]
        dropIdx = 0
        for i in range(len(FeatureDimensions)): # 4 Stages (one for each FeatureDimension) with multiple residual blocks of given respective depths.
            self.FeatureResolutionStages.append(ConvNeXtStage(Depths[i], 
                                                              FeatureDimensions[i], 4, 
                                                              dropIdx, self.dropPathRates,
                                                              LayerScaleInitial, 
                                                              Training))
            dropIdx += Depths[i]
            
        for i in range(len(FeatureDimensions)):
            self.add_module(f"LN{i}", LayerNorm(FeatureDimensions[i], _format="channels_first"))
        
    def forward(self, x):
        outputs = []
        for i in range(4): # 4 of each
            x = self.DownsampleLayers[i](x)
            x = self.FeatureResolutionStages[i](x)
            NL = getattr(self, f"LN{i}")
            outputs.append(NL(x))
        return outputs

class CBA(nn.Sequential):
    def __init__(self, in_Channels, out_Channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(nn.Conv2d(in_Channels, out_Channels, kernel_size, stride, padding, dilation, groups, bias=False),
                         nn.BatchNorm2d(out_Channels),
                         nn.ReLU(True))

class CBA_UP128(nn.Sequential):
    def __init__(self, in_Channels, out_Channels, groups = 1):
        super().__init__(nn.Conv2d(in_Channels, in_Channels // 4, 1, groups = groups),
                         nn.BatchNorm2d(in_Channels // 4),
                         nn.ReLU(True),
                         nn.ConvTranspose2d(in_Channels // 4, in_Channels // 4, 3, stride=2, padding=1, output_padding=1, groups = groups),
                         nn.BatchNorm2d(in_Channels // 4),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(in_Channels // 4, out_Channels, 1, groups = groups),
                         nn.BatchNorm2d(out_Channels),
                         nn.ReLU(inplace=True))

class CBA_UP256_r(nn.Sequential):
    def __init__(self, in_Channels, out_Channels, groups = 1):
        super().__init__(nn.ConvTranspose2d(in_Channels, out_Channels, 3, stride=2, groups = groups),
                         nn.ReLU(inplace=True))

class CBA_UP256_a(nn.Sequential):
    def __init__(self, in_Channels, out_Channels, groups = 1):
        super().__init__(nn.ConvTranspose2d(in_Channels, out_Channels, 3, stride=2, groups = groups),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_Channels, out_Channels, 3),
                         nn.ReLU(inplace=True))

class PPM(nn.Module):
    """Pyramid Pooling Module from PSPNet 
        https://arxiv.org/abs/1612.01105
    """
    def __init__(self, in_LastFeatureChannels, FPN_Dimension, scales):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(scale),
                                                   CBA(in_LastFeatureChannels, FPN_Dimension, 1)) for scale in scales])
        self.B = CBA(in_LastFeatureChannels + FPN_Dimension * len(scales), FPN_Dimension, 3, 1, 1)

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            outputs.append(F.interpolate(stage(x), size=x.shape[-2:], mode='bilinear', align_corners=True))
        outputs = [x] + outputs[::-1]
        output = self.B(torch.cat(outputs, dim=1))
        return output

class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 nn.BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out

class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
        From https://arxiv.org/abs/1909.06121
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = nn.BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = nn.BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out

class GraphReasoning(nn.Module):
    def __init__(self, FPN_Dimension, task_classes, ratio):
        super().__init__()
        self.DGCN257 = DualGCN(planes = 32, ratio = ratio)
        self.DGCN128 = DualGCN(planes = 32, ratio = ratio)
        self.DGCN64 = DualGCN(planes = 32, ratio = ratio)
        self.MaxPool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.Conv = nn.Conv2d(32, 32, 1)
        self.Conv256 = nn.Conv2d(32, 32, 3)
        self.Act = nn.ReLU(inplace=True)
        self.RoadToTaskClass = nn.Conv2d(32, task_classes, 1)
        self.Road256ToTaskClass = nn.Conv2d(32, task_classes, 2, padding=1)
        self.dropout = nn.Dropout2d(0.0)
        
    def forward(self, r_o64, r_o128, r_o257):
        o_gcn_257 = self.DGCN257(r_o257)
        o_gcn_128 = self.DGCN128(r_o128)
        o_gcn_64 = self.DGCN64(r_o64)
        
        o_gcn_64_to_128 = F.interpolate(o_gcn_64, size=(o_gcn_128.shape[2],o_gcn_128.shape[3]), mode="bilinear")
        o_gcn_128_and_64= torch.add(o_gcn_64_to_128, o_gcn_128)
        o_gcn_128_to_257  = F.interpolate(o_gcn_128_and_64, size=(o_gcn_257.shape[2],o_gcn_257.shape[3]), mode="bilinear")
        o_gcn256 = torch.add(o_gcn_128_to_257, o_gcn_257)
        
        o_gcn_conv64 = self.Conv(o_gcn_64)
        o_a64 = self.Act(o_gcn_conv64)
        o_gcn_conv128 = self.Conv(o_gcn_128_and_64)
        o_a128 = self.Act(o_gcn_conv128)
        o_gcn_conv256 = self.Conv256(o_gcn256)
        o_a256 = self.Act(o_gcn_conv256)
        
        outs = []
        outs.append(self.RoadToTaskClass(self.dropout(o_a64)))
        outs.append(self.RoadToTaskClass(self.dropout(o_a128)))
        outs.append(self.Road256ToTaskClass(self.dropout(o_a256)))
        return outs
    
class Decoder(nn.Module):
    """Adapted from a Unified Perceptual Parsing for Scene Understanding
    https://arxiv.org/abs/1807.10221
    Note that this implementation of UPerNet uses multiple scales for loss and GraphReasoning at the end.
    """
    
    def __init__(self, in_Channels, FPN_Dimension, tasks_classes, scales):
        super().__init__()
        self.tasks_classes = tasks_classes
        self.PPM = PPM(in_Channels[-1], FPN_Dimension, scales)
        self.FPN_in = nn.ModuleList()
        self.FPN_out = nn.ModuleList()
        for in_channels in in_Channels[:-1]:
            self.FPN_in.append(CBA(in_channels, FPN_Dimension, 1))
            self.FPN_out.append(CBA(FPN_Dimension, FPN_Dimension, 3, 1, 1))
        
        self.B1r = CBA(len(in_Channels)*FPN_Dimension, 32, 3, 1, 1)
        self.B1a = CBA(FPN_Dimension, FPN_Dimension, 3, 1, 1)
        self.B2r = CBA_UP128(32,32)
        self.B2a = CBA_UP128(FPN_Dimension,FPN_Dimension)
        self.B3r = CBA_UP256_r(32,32)
        self.B3a = CBA_UP256_a(FPN_Dimension, 32)
        self.dropout = nn.Dropout2d(0.0)
        self.AngleToTaskClass = nn.Conv2d(FPN_Dimension, 37, 1)
        self.Angle257ToTaskClass = nn.Conv2d(32, 37, 2, padding = 1)
        self.GraphReasoning = GraphReasoning(FPN_Dimension, 2, ratio = 2)
            
    def forward(self, features):
        FPN_feature_level = self.PPM(features[-1])
        
        FPN_features = [FPN_feature_level]
        for i in reversed(range(len(features)-1)):
            feature = self.FPN_in[i](features[i])
            FPN_feature_level = feature + F.interpolate(FPN_feature_level, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            FPN_features.append(self.FPN_out[i](FPN_feature_level))

        FPN_features.reverse()
        P2 = FPN_features[0] # for angle
        for i in range(1, len(features)):
            FPN_features[i] = F.interpolate(FPN_features[i], size=FPN_features[0].shape[-2:], mode='bilinear', align_corners=False)
        
        road_o64 = self.B1r(torch.cat(FPN_features, dim=1)) # ([16, 128, 64, 64]) # fusing part for road
        road_o128 = self.B2r(road_o64) # ([16, 128, 128, 128])
        road_o257 = self.B3r(road_o128) # ([16, 32, 257, 257])
        
        angle_o64 = self.B1a(P2)
        angle_o128 = self.B2a(angle_o64)
        angle_o257 = self.B3a(angle_o128)
        
        outputs = [[] for x in range(len(self.tasks_classes))]
        
        road_task_outputs = self.GraphReasoning(road_o64, road_o128, road_o257)
        
        angle_task_output64 = self.AngleToTaskClass(self.dropout(angle_o64))
        angle_task_output128 = self.AngleToTaskClass(self.dropout(angle_o128))
        angle_task_output256 = self.Angle257ToTaskClass(self.dropout(angle_o257))
        
        outputs[0].append(road_task_outputs[0]) # 64
        outputs[0].append(road_task_outputs[1]) # 128
        outputs[0].append(road_task_outputs[2]) # 256
        outputs[1].append(angle_task_output64)
        outputs[1].append(angle_task_output128)
        outputs[1].append(angle_task_output256)
        
        return outputs

class ConvNeXt_UPerNet_DGCN_MTL(nn.Module):
    def __init__(self, ModelArch = "Base"):
        super().__init__()
        assert ModelArch in ConvNeXt_Archs.keys()
        Depths, FeatureDimensions, StochasticDepthRate, LayerScaleInitial = ConvNeXt_Archs[ModelArch] # 'Base': [[3, 3, 27, 3], [128, 256, 512, 1024], 0.4, 1.0],
        
        self.encoder = Encoder(3, Depths, FeatureDimensions, StochasticDepthRate, LayerScaleInitial, Training = self.training)
        self.decoder = Decoder(FeatureDimensions, FPN_Dimension = FeatureDimensions[0], tasks_classes = [2,37], scales = (1, 2, 3, 6)) # instead of 1,2,3,6 (arcgis (1, 3, 5, 6))
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x