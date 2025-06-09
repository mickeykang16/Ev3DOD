import os
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv

from al3d_utils import common_utils
from al3d_utils.ops.iou3d_nms import iou3d_nms_utils

from al3d_det.models import fusion_modules
from .centerpoint_waymo import CenterPointPC
from al3d_det.utils import nms_utils
from al3d_det.models import image_modules as img_modules
from al3d_det.models import modules as cp_modules
import pdb

from al3d_det.utils.visual_utils.vis_data import save_data

# import torch.nn.functional as f

from al3d_utils.timers import TimerDummy as CudaTimer
# from al3d_utils.timers import CudaTimer


class CenterPointMM_LiDAR(CenterPointPC):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)

    def forward(self, batch_dict, cur_module=None, end=False, ret_lidar=None, index=None):
        if not end:
            if index is not None:
                return cur_module(batch_dict, index)
            else:
                return cur_module(batch_dict)
        else:
            if self.training:
                loss, tb_dict, disp_dict = self.get_training_loss(index)
                if ret_lidar is not None:
                    loss += ret_lidar[0]['loss']
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict, index)
                return pred_dicts, recall_dicts

class CenterPointMM_Camera(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.img_backbone = img_modules.__all__[model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=model_cfg.IMAGE_BACKBONE
        )
        self.ev_backbone = img_modules.__all__[model_cfg.EVENT_BACKBONE.NAME](
            model_cfg=model_cfg.EVENT_BACKBONE
        )
        
        # self.ei_fusion = img_modules.__all__[model_cfg.EI_FUSION.NAME](
        #     model_cfg=model_cfg.EI_FUSION
        # )
        
        if 'IMGPRETRAINED_MODEL' in model_cfg.IMAGE_BACKBONE and model_cfg.IMAGE_BACKBONE.IMGPRETRAINED_MODEL is not None:
            checkpoint= torch.load(model_cfg.IMAGE_BACKBONE.IMGPRETRAINED_MODEL, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            ckpt = state_dict
            new_ckpt = OrderedDict()
            for k, v in ckpt.items():
                if k.startswith('backbone'):
                    new_v = v
                    new_k = k.replace('backbone.', 'img_backbone.')
                else:
                    continue
                new_ckpt[new_k] = new_v
            self.img_backbone.load_state_dict(new_ckpt, strict=False)


class CenterPointMM(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        
        self.lidar = CenterPointMM_LiDAR(
            model_cfg=model_cfg, num_class=num_class, dataset=dataset)
 
        self.camera = CenterPointMM_Camera(model_cfg)
        self.training = self.lidar.training
        self.second_stage = self.lidar.second_stage
        self.pretrain = self.lidar.pretrain
        
        self.grid_size = self.lidar.dataset.grid_size[::-1] + [1, 0, 0]
        voxel_size  = self.lidar.dataset.voxel_size
        point_cloud_range = self.lidar.dataset.point_cloud_range
        
        
        if not self.pretrain:
            for param in self.camera.img_backbone.parameters():
                param.requires_grad = False
            for param in self.lidar.module_list[0].parameters():
                param.requires_grad = False
            for param in self.lidar.module_list[1].parameters():
                param.requires_grad = False
            for param in self.lidar.module_list[2].parameters():
                param.requires_grad = False
            for param in self.lidar.module_list[3].parameters():
                param.requires_grad = False
            for param in self.lidar.module_list[4].parameters():
                param.requires_grad = False
            for param in self.lidar.module_list[5].parameters():
                param.requires_grad = False
            # pdb.set_trace()
            
            
            if model_cfg.ROI_HEAD_EV.get('TRAIN_EV_CLS_ONLY', False):
                for param in self.lidar.module_list[6].parameters():
                    param.requires_grad = False
                for param in self.lidar.module_list[6].ev_cls_layers.parameters():
                    param.requires_grad = True
        
        # for name, param in self.camera.named_parameters():
        #     print(name, param.requires_grad)
        print("########################################################")
        for name, param in self.lidar.named_parameters():
            if param.requires_grad == True:
                print(name)
        print("########################################################")
        # self.freeze_img = model_cfg.IMAGE_BACKBONE.get('FREEZE_IMGBACKBONE', False)
        # self.freeze()
        
        
    def freeze(self):
        if self.freeze_img:
            for param in self.camera.img_backbone.img_backbone.parameters():
                param.requires_grad = False

            for param in self.camera.img_backbone.neck.parameters():
                param.requires_grad = False
    def forward(self, batch_dict):
        # voxel coord [0, Z, Y, X] <= typical lidar frame 
        
        self.camera.img_backbone.eval()
        self.lidar.module_list[0].eval()
        self.lidar.module_list[1].eval()
        self.lidar.module_list[2].eval()
        self.lidar.module_list[3].eval()
        self.lidar.module_list[4].eval()
        self.lidar.module_list[5].eval()
        # self.lidar.module_list[10].eval()
        self.camera.ev_backbone.eval()
        
        
        ret_lidar = None
        # pdb.set_trace()
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[0])
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[1])
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[2])
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[3])
        batch_dict = self.camera.img_backbone(batch_dict)
        if not self.training:
            ret_lidar_list = []
            
        
        # import pdb; pdb.set_trace()
        # pcd = batch_dict['points'][:, -3:][batch_dict['points'][:, 0] == 0].detach().cpu().numpy()
        
        # image = f.interpolate(batch_dict['images']['camera_0'], size = (1280, 1920))[0].detach().cpu().numpy()
        # extr = batch_dict['extrinsic']['camera_0'][0].detach().cpu().numpy()
        # intr = batch_dict['intrinsic']['camera_0'][0].detach().cpu().numpy()
        # events = f.interpolate(batch_dict['events'][0], size = (1280, 1920)).detach().cpu().numpy()
        # gt_boxes = [box[0].detach().cpu().numpy() for box in batch_dict['gt_boxes']]
        # pdb.set_trace()
        # aug_mat  = batch_dict.get('aug_matrix_inv', None)
        # if aug_mat is not None:
        #     aug_mat = aug_mat[0]
        
        # save_data({
        #     'pcd': pcd,
        #     'image': image,
        #     'extr': extr,
        #     'intr': intr,
        #     'events': events,
        #     'gt_boxes': gt_boxes,
        #     'aug_mat': aug_mat
        # })
        
        batch_dict = self.lidar(batch_dict, self.lidar.module_list[4], index=0)
        
        # torch.cuda.empty_cache()
        
        for index in range(10):
            # batch_dict = self.lidar(batch_dict, self.lidar.module_list[4], index=index)
            if index == 0:
                if self.second_stage:
                    batch_dict = self.lidar(batch_dict, self.lidar.module_list[5], index=index)
                if not self.training:
                    ret_lidar = self.lidar(batch_dict, end=True, ret_lidar=ret_lidar, index=index)
                    
                    max_ = 0
                    for batch_pred in ret_lidar[0]:
                        cur_len = batch_pred['pred_boxes'].shape[0]
                        if cur_len > max_:
                            max_ = cur_len
                    pred_boxes_list = []
                    pred_scores_list = []
                    pred_labels_list = []
                    for batch_pred in ret_lidar[0]:    
                        pred_boxes_ = torch.zeros((max_, batch_pred['pred_boxes'].shape[1]), device=batch_pred['pred_boxes'].device)
                        pred_scores_ = torch.zeros((max_), device=batch_pred['pred_boxes'].device)
                        pred_labels_ = torch.zeros((max_), device=batch_pred['pred_boxes'].device)
                        
                        pred_boxes_[:batch_pred['pred_boxes'].shape[0], :] = batch_pred['pred_boxes']
                        pred_scores_[:batch_pred['pred_scores'].shape[0]] = batch_pred['pred_scores']
                        pred_labels_[:batch_pred['pred_labels'].shape[0]] = batch_pred['pred_labels']
                        
                        pred_boxes_list.append(pred_boxes_)
                        pred_scores_list.append(pred_scores_)
                        pred_labels_list.append(pred_labels_)
                    
                    pred_boxes = torch.stack(pred_boxes_list, dim=0).clone()
                    pred_scores = torch.stack(pred_scores_list, dim=0).clone()
                    pred_labels = torch.stack(pred_labels_list, dim=0).clone()
                    # pdb.set_trace()
                    batch_dict['rois'] = pred_boxes
                    batch_dict['roi_scores'] = pred_scores
                    batch_dict['roi_labels'] = pred_labels.long()
                else:
                    if self.pretrain:
                        ret_lidar = self.lidar(batch_dict, end=True, ret_lidar=ret_lidar, index=index)
                if self.pretrain:
                    if self.training:
                        return ret_lidar
                    else:
                        return [ret_lidar]
            else:
                batch_dict = self.camera.ev_backbone(batch_dict, index)
                
                # pdb.set_trace()
                # batch_dict['event_features']['layer1_feat2d'].shape 1x32x160x240
                    
                batch_dict = self.lidar(batch_dict, self.lidar.module_list[6], index=index)
                ret_lidar = self.lidar(batch_dict, end=True, ret_lidar=ret_lidar, index=index)
                
                
                # if not self.training:
                #     max_ = 0
                #     for batch_pred in ret_lidar[0]:
                #         cur_len = batch_pred['pred_boxes'].shape[0]
                #         if cur_len > max_:
                #             max_ = cur_len
                #     pred_boxes_list = []
                #     pred_scores_list = []
                #     pred_labels_list = []
                #     for batch_pred in ret_lidar[0]:    
                #         pred_boxes_ = torch.zeros((max_, batch_pred['pred_boxes'].shape[1]), device=batch_pred['pred_boxes'].device)
                #         pred_scores_ = torch.zeros((max_), device=batch_pred['pred_boxes'].device)
                #         pred_labels_ = torch.zeros((max_), device=batch_pred['pred_boxes'].device)
                        
                #         pred_boxes_[:batch_pred['pred_boxes'].shape[0], :] = batch_pred['pred_boxes']
                #         pred_scores_[:batch_pred['pred_scores'].shape[0]] = batch_pred['pred_scores']
                #         pred_labels_[:batch_pred['pred_labels'].shape[0]] = batch_pred['pred_labels']
                        
                #         pred_boxes_list.append(pred_boxes_)
                #         pred_scores_list.append(pred_scores_)
                #         pred_labels_list.append(pred_labels_)
                    
                #     pred_boxes = torch.stack(pred_boxes_list, dim=0).clone()
                #     pred_scores = torch.stack(pred_scores_list, dim=0).clone()
                #     pred_labels = torch.stack(pred_labels_list, dim=0).clone()
                #     batch_dict['rois'] = pred_boxes
                #     batch_dict['roi_scores'] = pred_scores
                #     batch_dict['roi_labels'] = pred_labels.long()
                
            
            # if index == 0:
            #     continue
            # pdb.set_trace()
            
                
            
            # if not (batch_dict['rois'].shape[1] == batch_dict['roi_labels'].shape[1]  \
            # and batch_dict['roi_labels'].shape[1] == batch_dict['roi_scores'].shape[1] \
            #     and batch_dict['rois'].shape[1] == batch_dict['roi_scores'].shape[1]):
            #     pdb.set_trace()
            if not self.training:
                ret_lidar_list.append(ret_lidar)
        
        # pdb.set_trace()
        save_dir='../output/viz_feat/' + str(batch_dict['sequence_name'][0])
        os.makedirs(save_dir, exist_ok=True)
        for i, feat_dict in enumerate(batch_dict['viz_feat']):
            file_name = os.path.join(save_dir, str(batch_dict['frame_id'][0]).zfill(4) + f'_{i+1}.npz')
            np.savez(file_name, 
                     ev_feat=feat_dict['ev_feat'][0],
                     ev_voxel_feat=feat_dict['ev_voxel_feat'],
                     point_features=feat_dict['point_features'],
                     point_coords=feat_dict['point_coords'][:, 1:],)
        
        if self.training:
            return ret_lidar
        else:
            # return ret_lidar
            return ret_lidar_list

    def update_global_step(self):
        if hasattr(self.lidar, 'update_global_step'):
            self.lidar.update_global_step()
        else:
            self.module.lidar.update_global_step()

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' %
                    (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' %
                        checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' %
                            (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' %
                    (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' %
                    (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(
                        optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(
                        optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' %
                  checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    
