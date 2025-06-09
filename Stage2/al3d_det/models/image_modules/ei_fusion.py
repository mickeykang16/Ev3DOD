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

class EIFUSION(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        feat_dim = model_cfg.EMBED_DIM
        self.conv1 = nn.Sequential(nn.Conv2d(feat_dim*2, feat_dim, kernel_size=(3, 3), padding=(1,1)),                                                                                                                                               
                                    nn.BatchNorm2d(feat_dim))
                                    # nn.LayerNorm((64,), eps=1e-05, elementwise_affine=True))

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

        try:
            data = torch.cat([batch_dict['image_features']['layer1_feat2d']['camera_0'], 
                          batch_dict['event_features']['layer1_feat2d']], dim=1)
        except:
            pdb.set_trace()
        bs = batch_dict['batch_size']
        
        fused_features = self.conv1(data)
        
        batch_dict['image_features']['layer1_feat2d']['camera_0'] = fused_features
        return batch_dict
