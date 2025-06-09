from mmdet.models.builder import build_backbone, build_neck
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from kornia import normalize
except:
    pass
from collections import OrderedDict
from al3d_det.models.image_modules.ifn.basic_blocks import BasicBlock2D
import pdb

class MMDETFPN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.imgconfig = model_cfg.IMGCONFIG
        self.fpnconfig = model_cfg.FPNCONFIG
        self.img_backbone = build_backbone(self.imgconfig)
        if self.model_cfg.get('PRETRAINEDPATH', None) is not None:
            self.img_backbone.init_weights(self.model_cfg.PRETRAINEDPATH)
        if self.imgconfig.get('pretrained', None) is not None: 
            self.img_backbone.init_weights(self.imgconfig.pretrained)

        self.neck = build_neck(self.fpnconfig)
        self.reduce_blocks = torch.nn.ModuleList()
        self.out_channels = {}
        for _idx, _channel in enumerate(model_cfg.IFN.CHANNEL_REDUCE["in_channels"]):
            _channel_out = model_cfg.IFN.CHANNEL_REDUCE["out_channels"][_idx]
            self.out_channels[model_cfg.IFN.ARGS['feat_extract_layer'][_idx]] = _channel_out
            block_cfg = {"in_channels": _channel,
                         "out_channels": _channel_out,
                         "kernel_size": model_cfg.IFN.CHANNEL_REDUCE["kernel_size"][_idx],
                         "stride": model_cfg.IFN.CHANNEL_REDUCE["stride"][_idx],
                         "bias": model_cfg.IFN.CHANNEL_REDUCE["bias"][_idx]}
            self.reduce_blocks.append(BasicBlock2D(**block_cfg))

    def get_output_feature_dim(self):
        return self.out_channels
    def forward(self, batch_dict):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # Preprocess images
        # x = self.preprocess(images)

        # Extract features
        result = OrderedDict()
        data = batch_dict['images']
        bs = batch_dict['batch_size']
        
        batch_dict['image_features'] = {}
        image_list = []
        
        for cam in data.keys():
            single_result = {}
            image_list.append(data[cam].unsqueeze(1))
        images = torch.cat(image_list, dim=1)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)
        x = self.img_backbone(images)
        x_neck = self.neck(x)
        for _idx, _layer in enumerate(self.model_cfg.IFN.ARGS['feat_extract_layer']):
            image_features = x_neck[_idx]
            if self.reduce_blocks[_idx] is not None:
                image_features = self.reduce_blocks[_idx](image_features)
            single_result[_layer+"_feat2d"] = image_features
        for layer in single_result.keys():
            if layer not in batch_dict['image_features'].keys():
                batch_dict['image_features'][layer] = {}
            for i in range(len(data.keys())):
                cam = 'camera_{}'.format(i)
                batch_dict['image_features'][layer][cam] = single_result[layer][i*B:(i+1)*B]
        return batch_dict

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        # if self.pretrained:
        #     # Create a mask for padded pixels
        #     mask = torch.isnan(x)

        #     # Match ResNet pretrained preprocessing
        #     x = normalize(x, mean=self.norm_mean, std=self.norm_std)

        #     # Make padded pixels = 0
        #     x[mask] = 0

        return x

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer    
    
class MMDETFPNEVDOWN(nn.Module):
    def __init__(self, model_cfg, in_size=5, out_size=64, downsample=True, relu_slope=0.2, use_emgc=False):
        super().__init__()
        self.model_cfg = model_cfg
        out_size = self.model_cfg.OUT_CHANNELS
        self.downsample = downsample
        self.identity = nn.Conv2d(out_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.downsample1 = conv_down(out_size, out_size, bias=False)

        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.downsample2 = conv_down(out_size, out_size, bias=False)

        # self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0) 
       
        
    def forward(self, batch_dict, index):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # Preprocess images
        # x = self.preprocess(images)

        # Extract features
        result = OrderedDict()
        data = batch_dict['events'][:, index-1]
        bs = batch_dict['batch_size']
        
        batch_dict['event_features'] = {}
        
        single_result = {}
        
        out = self.conv_1(data)
        out_conv1 = self.relu_1(out)
        
        out_down1 = self.downsample1(out_conv1)
        
        out_conv2 = self.relu_2(self.conv_2(out_down1))

        out = out_conv2 + self.identity(out_down1)
        
        
        image_features = self.downsample2(out)
        
        
        # out_down = self.downsample2(out)
        # image_features = self.conv_before_merge(out_down)
        
        
        # x = self.img_backbone(data)
        # x_neck = self.neck(x)
        for _idx, _layer in enumerate(self.model_cfg.IFN.ARGS['feat_extract_layer']):
            # image_features = x_neck[_idx]
            # if self.reduce_blocks[_idx] is not None:
            #     image_features = self.reduce_blocks[_idx](image_features)
            single_result[_layer+"_feat2d"] = image_features
        
        for layer in single_result.keys():
            # if layer not in batch_dict['event_features'].keys():
            #     batch_dict['event_features'][layer] = {}
            batch_dict['event_features'][layer] = single_result[layer]
        return batch_dict

  


    
class MMDETFPNEV(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.imgconfig = model_cfg.IMGCONFIG
        self.fpnconfig = model_cfg.FPNCONFIG
        self.img_backbone = build_backbone(self.imgconfig)
        if self.model_cfg.get('PRETRAINEDPATH', None) is not None:
            self.img_backbone.init_weights(self.model_cfg.PRETRAINEDPATH)
        if self.imgconfig.get('pretrained', None) is not None: 
            self.img_backbone.init_weights(self.imgconfig.pretrained)

        self.neck = build_neck(self.fpnconfig)
        self.reduce_blocks = torch.nn.ModuleList()
        self.out_channels = {}
        for _idx, _channel in enumerate(model_cfg.IFN.CHANNEL_REDUCE["in_channels"]):
            _channel_out = model_cfg.IFN.CHANNEL_REDUCE["out_channels"][_idx]
            self.out_channels[model_cfg.IFN.ARGS['feat_extract_layer'][_idx]] = _channel_out
            block_cfg = {"in_channels": _channel,
                         "out_channels": _channel_out,
                         "kernel_size": model_cfg.IFN.CHANNEL_REDUCE["kernel_size"][_idx],
                         "stride": model_cfg.IFN.CHANNEL_REDUCE["stride"][_idx],
                         "bias": model_cfg.IFN.CHANNEL_REDUCE["bias"][_idx]}
            self.reduce_blocks.append(BasicBlock2D(**block_cfg))

        self.feat_dim_config = self.imgconfig.embed_dim
        self.img_backbone.patch_embed.proj = nn.Conv2d(5, self.feat_dim_config, kernel_size=(4,4), stride=(4,4))
        
    def get_output_feature_dim(self):
        return self.out_channels
    def forward(self, batch_dict, index):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # Preprocess images
        # x = self.preprocess(images)

        # Extract features
        result = OrderedDict()
        data = batch_dict['events'][:, index-1]
        bs = batch_dict['batch_size']
        
        batch_dict['event_features'] = {}
        
        single_result = {}
        
        x = self.img_backbone(data)
        x_neck = self.neck(x)
        for _idx, _layer in enumerate(self.model_cfg.IFN.ARGS['feat_extract_layer']):
            image_features = x_neck[_idx]
            if self.reduce_blocks[_idx] is not None:
                image_features = self.reduce_blocks[_idx](image_features)
            single_result[_layer+"_feat2d"] = image_features
        
        for layer in single_result.keys():
            # if layer not in batch_dict['event_features'].keys():
            #     batch_dict['event_features'][layer] = {}
            batch_dict['event_features'][layer] = single_result[layer]
        return batch_dict

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        # if self.pretrained:
        #     # Create a mask for padded pixels
        #     mask = torch.isnan(x)

        #     # Match ResNet pretrained preprocessing
        #     x = normalize(x, mean=self.norm_mean, std=self.norm_std)

        #     # Make padded pixels = 0
        #     x[mask] = 0

        return x