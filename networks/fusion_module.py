import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *
from .IFRNet import warp

class Embedder(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], 1)
            

class FusionModule(nn.Module):
    def __init__(self, args, num_ch_enc, embed_multires=10):
        super(FusionModule, self).__init__()

        embed_kwargs = {
                    "include_input": True,
                    "input_dims": 2,
                    "max_freq_log2": embed_multires - 1,
                    "num_freqs": embed_multires,
                    "log_sampling": True,
                    "periodic_fns": [torch.sin, torch.cos],
        }

        self.embedder_obj = Embedder(**embed_kwargs)
        self.embed_multires = embed_multires
        self.num_ch_enc = num_ch_enc
        self.backbone = args.backbone

        self.convs = OrderedDict()

        for i in range(len(num_ch_enc)-1, -1, -1):
            self.convs[("conv1x1", i)] = ConvBlock1x1(2*(num_ch_enc[i]+self.embedder_obj.out_dim), num_ch_enc[i])

        self.fusion_conv = nn.ModuleList(list(self.convs.values()))

    def get_embedding_flow(self, x):
        oups = []
        for i in range(len(self.num_ch_enc)):
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            x[:, 0, :, :] *= 0.5
            x[:, 1, :, :] *= 0.5           
            if i == 0 and self.backbone == "LiteMono":
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                x[:, 0, :, :] *= 0.5
                x[:, 1, :, :] *= 0.5 
            y = self.embedder_obj.embed(x)
            oups.append(y)     

        return oups 

    def warp_features(self, features, flow):
        feats_warp = []
        for feat in features:
            _, _, flow_h, flow_w = flow.shape
            _, _, H, W = feat.shape
            flow_ = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
            flow_[:, 0, :, :] *= (W / flow_w)
            flow_[:, 1, :, :] *= (H / flow_h)
            feat_warp = warp(feat, flow_)
            feats_warp.append(feat_warp)
        return feats_warp

    def merge_features(self, features_warped, merge_mask):
        feats_n1, feats_0, feats_p1 = features_warped
        mask = merge_mask
        feats = []
        for i in range(len(self.num_ch_enc)):
            feat_n1, feat_0, feat_p1 = feats_n1[i], feats_0[i], feats_p1[i]
            _, _, H, W = feat_0.shape
            mask_ = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
            feat = mask_ * feat_n1 + (1-mask_) * feat_p1
            feats.append(torch.cat([feat_0, feat], dim=1))

        return feats
    
    def forward(self, features, flows, merge_mask):
        feats_n1, feats_0, feats_p1 = features
        flow_0_n1, flow_0_p1 = flows

        feats_n1_0 = self.warp_features(feats_n1, flow_0_n1)
        feats_p1_0 = self.warp_features(feats_p1, flow_0_p1)
        feats_0_0 = [0 for _ in range(len(feats_0))]
        
        # flow_0 = torch.zeros_like(flow_0_n1).to(flow_0_n1.device)
        flow_0 = 0. * flow_0_n1.clone().detach()
        emb_flows_0 = self.get_embedding_flow(flow_0)
        emb_flows_0_n1 = self.get_embedding_flow(flow_0_n1)
        emb_flows_0_p1 = self.get_embedding_flow(flow_0_p1)

        for i in range(len(feats_0)):
            feats_0_0[i] = torch.cat([feats_0[i], emb_flows_0[i]], 1)
            feats_n1_0[i]= torch.cat([feats_n1_0[i], emb_flows_0_n1[i]], 1)
            feats_p1_0[i] = torch.cat([feats_p1_0[i], emb_flows_0_p1[i]], 1)

        all_feats = [feats_n1_0, feats_0_0, feats_p1_0]
        all_feats = self.merge_features(all_feats, merge_mask)

        for i in range(len(all_feats)):
            all_feats[i] = self.convs[("conv1x1", i)](all_feats[i])

        return all_feats


    