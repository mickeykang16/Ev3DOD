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

class GOF_LSTM(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        feat_dim = model_cfg.EMBED_DIM
        # self.conv1 = nn.Sequential(nn.Conv2d(feat_dim*2, feat_dim, kernel_size=(3, 3), padding=(1,1)),                                                                                                                                               
        #                             nn.BatchNorm2d(feat_dim))
                                    # nn.LayerNorm((64,), eps=1e-05, elementwise_affine=True))
        
        self.lstm = CLSTM_cell( 
            input_channels=feat_dim, 
            filter_size=3, 
            num_features=feat_dim
        )
        
        self.conv1 = nn.Sequential(nn.Conv2d(feat_dim*2, feat_dim, kernel_size=(3, 3), padding=(1,1)),                                                                                                                                               
                            nn.BatchNorm2d(feat_dim))
                                    
                                    

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

        # data = torch.cat([batch_dict['image_features']['layer1_feat2d']['camera_0'], 
        #                   batch_dict['event_features']['layer1_feat2d']], dim=1)
        
        data =  batch_dict['event_features']['layer1_feat2d']
        img_feat = batch_dict['image_features']['layer1_feat2d']['camera_0']
        bs = batch_dict['batch_size']
        
        if 'gof_hidden_state' in batch_dict['event_features'].keys():
            hidden_state = self.lstm(data, batch_dict['event_features']['gof_lstm_feat'])
        else:
            hidden_state = self.lstm(data)
        
        fused_feat =  self.conv1(torch.cat([img_feat, hidden_state[0]], dim=1))
        
        batch_dict['event_features']['gof_lstm_feat'] = hidden_state
        batch_dict['event_features']['gof_hidden_state'] = [fused_feat]
        
        
        
        
        # batch_dict['GoF_result_feats']
        # pdb.set_trace()
        
        return batch_dict


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

         # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        # self.conv = nn.Sequential(
        #     nn.Conv2d(self.input_channels + self.num_features,
        #               4 * self.num_features, self.filter_size, 1,
        #               self.padding),
        #     nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))
        
        self.conv = nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding)
        

    def forward(self, inputs, hidden_state=None):
        
        if hidden_state is None:
            hx = torch.zeros(inputs.shape[0], self.num_features, inputs.shape[2],
                             inputs.shape[3]).cuda()
            cx = torch.zeros(inputs.shape[0], self.num_features, inputs.shape[2],
                             inputs.shape[3]).cuda()
        else:
            hx, cx = hidden_state
        
     
        x = inputs


        combined = torch.cat((x, hx), 1)
        gates = self.conv(combined)  # gates: S, num_features*4, H, W
        # it should return 4 tensors: i,f,g,o
        ingate, forgetgate, cellgate, outgate = torch.split(
            gates, self.num_features, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        hx = hy
        cx = cy
        
        return hy, cy
        
        # return torch.stack(output_inner), (hy, cy)