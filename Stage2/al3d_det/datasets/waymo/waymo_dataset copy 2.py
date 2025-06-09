import os
import pickle5 as pickle
import io
import copy
import re

import numpy as np
import cv2
# from petrel_client.client import Client
from al3d_utils import common_utils
from al3d_utils.ops.roiaware_pool3d import roiaware_pool3d_utils
from al3d_utils.aws_utils import list_oss_dir, oss_exist

from al3d_det.datasets.dataset import DatasetTemplate
from al3d_det.datasets.augmentor.data_augmentor import DataAugmentor
from al3d_det.datasets.augmentor.test_time_augmentor import TestTimeAugmentor

import torch.distributed as dist
import pdb
import torch.nn.functional as F
import torch

class WaymoInferenceDataset(DatasetTemplate):
    """
    The Dataset class for Inference on Waymo
    """
    def __init__(self, dataset_cfg, class_names, data_infos, point_list, training=False, logger=None) -> None:
        super().__init__(dataset_cfg, class_names, training, logger)
        self.data_infos = data_infos
        self.point_list = point_list
        self.init_infos()

    def init_infos(self):
        self.infos = self.data_infos

    def get_infos_and_points(self, idx_list):
        infos, points = [], []
        for i in idx_list:
            infos.append(self.infos[i])
            points.append(self.point_list[i])
        return infos, points


class WaymoTrainingDataset(DatasetTemplate):
    """
    The Dataset class for Training on Waymo (from File System)
    """

    # def __init__(self, dataset_cfg, class_names, root_path, training=True, logger=None) -> None:
    #     super().__init__(dataset_cfg, class_names, training, root_path, logger)
    #     self.data_path = self.root_path + '/' + dataset_cfg.PROCESSED_DATA_TAG
    #     self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
    #     split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '_sample.txt').replace('/ssd1tb_jy_3', '/home/user/jaeyoung/data2')
    #     # pdb.set_trace()
    #     if 's3' in self.root_path:
    #         from petrel_client.client import Client
    #         self.client = Client('~/.petreloss.conf')
    #         self.oss_data_list = list_oss_dir(self.data_path, self.client, with_info=False)
    #         self.sample_sequence_list = [x.decode().strip() for x in io.BytesIO(self.client.get(split_dir)).readlines()]
    #     else:
    #         self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
    #     # pdb.set_trace()
    #     self.init_infos()
    def __init__(self, dataset_cfg, class_names, root_path, training=True, logger=None) -> None:
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.data_path = self.root_path + '/' + dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        if self.mode == 'train':
            split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '_lost_bbox_20m.txt')        
        else:
            # split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '_lost_bbox_20m.txt')
            # split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '_sample.txt')
            split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '_viz.txt')
        if 's3' in self.root_path:
            from petrel_client.client import Client
            self.client = Client('~/.petreloss.conf')
            self.oss_data_list = list_oss_dir(self.data_path, self.client, with_info=False)
            self.sample_sequence_list = [x.decode().strip() for x in io.BytesIO(self.client.get(split_dir)).readlines()]
        else:
            
            self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
            
        self.init_infos()
    

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names,
            training=self.training, root_path=self.root_path,
            logger=self.logger
        )
        self.split = split
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '_sample.txt')
        if 's3' in self.root_path:
            self.sample_sequence_list = [x.decode().strip() for x in io.BytesIO(self.client.get(split_dir )).readlines()]
        else:
            self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.init_infos()

    def init_infos(self):
        
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
        # for k in range(10):
            if self.mode == 'train':
                # info_path = os.path.join(self.data_path, sequence_name, ('%s.pkl' % sequence_name)).replace('.pkl', '_fov_bbox.pkl')
                match_num = int(self.sample_sequence_list[k].split(', ')[0])
                if match_num == 0:
                    continue
                sequence_name = self.sample_sequence_list[k].split(', ')[1]
                info_path = os.path.join(self.data_path, sequence_name, ('%s.pkl' % sequence_name)).replace('.pkl', '_interpolate_fov_bbox_lidar_check.pkl')      
            else:
                ################## for lost bbox eval ###########
                # match_num = int(self.sample_sequence_list[k].split(', ')[0])
                # if match_num < 10:
                #     continue
                # sequence_name = self.sample_sequence_list[k].split(', ')[1]
                #########################################
                sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
                # info_path = os.path.join(self.data_path, sequence_name, ('%s.pkl' % sequence_name)).replace('.pkl', '_fov_bbox.pkl')    
                info_path = os.path.join(self.data_path, sequence_name, ('%s.pkl' % sequence_name)).replace('.pkl', '_interpolate_fov_bbox_lidar_check.pkl')      
            info_path = self.check_sequence_name_with_all_version(info_path)
            if 's3' in self.root_path:
                if not self.client.contains(info_path):
                    num_skipped_infos += 1
                    continue
                pkl_bytes = self.client.get(info_path)
                infos = pickle.load(io.BytesIO(pkl_bytes))
            else:
                
                if not os.path.exists(info_path):
                    num_skipped_infos += 1
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    
          
                
            ######## 100 FPS
            if self.mode == 'train':
                new_infos = []
                # infos = infos[:-1]
                infos = infos[:100]
                
                match_number = [int(x) for x in self.sample_sequence_list[k].split(', ')[2:]]
                for i in range(len(infos)):
                    # if not i in match_number:
                    #     continue

                    new_info = copy.deepcopy(infos[i])
                    new_annos = []
                    event_paths = []
                    img_infos = infos[i]['image']
                    lidar_path = infos[i]['lidar_path']
                    # pdb.set_trace()
                    event_init_path = img_infos['image_' + str(i) + '_path']\
                                    .replace('/image_0', '')
                        # .replace('/mnt1/LoGoNet/detection/data/waymo/waymo_processed_data_v4', '/home/user/jaeyoung/data5/waymo_voxels_100FPS_from_start')\
        
                    # pdb.set_trace()
                    event_dir = '/' + os.path.join(*event_init_path.split('/')[:-1])
                    init_number = int(event_init_path.split('/')[-1].split('.png')[0]) + 1
                    
                    
                    for j in range(9):
                        event_paths.append(os.path.join(event_dir, str(init_number).zfill(4) + '_' + str(j+1) + '.npz'))
                    # pdb.set_trace()
                    event_paths = np.array(event_paths)
                    new_info['event_paths'] = event_paths
                    
                    new_info['seq_annos'] = np.array(infos[i]['annos'])
                    
                    ## change image and lidar path
 
                    #################################
                    # pdb.set_trace()
                    new_infos.append(new_info)
                infos = new_infos
            
            else:
                new_infos = []
                infos = infos[:-1] # TODO 
                # infos = infos[:-180]
                # infos = infos[::100]
                # infos = infos[0::10]
                # match_number = [int(x) for x in self.sample_sequence_list[k].split(', ')[2:]]
                
                for i in range(len(infos)):
                    # if not i in match_number:
                    #     continue
                    
                    new_info = copy.deepcopy(infos[i])
                    new_annos = []
                    event_paths = []
                    # new_info['annos'] = infos[i]['annos'][j]
                    img_infos = infos[i]['image']
                    lidar_path = infos[i]['lidar_path']
                    
                    try:
                        for key in img_infos.keys():
                            if 'path' not in key: continue
                        img_path = img_infos[key]
                        # event_init_path = img_infos['image_' + str(i) + '_path'].replace('/LoGoNet/detection/data/waymo/waymo_processed_data_v4', '/waymo_voxels_val')\
                        #             .replace('/image_0', '')
                        event_init_path = img_path\
                            .replace('/mnt1/LoGoNet/detection/data/waymo/waymo_processed_data_v4', '/mnt_temp/waymo_voxels_300FPS_from_start')\
                                    .replace('/image_0', '')
                    
                    except:
                        pdb.set_trace()
                    event_dir = '/' + os.path.join(*event_init_path.split('/')[:-1])
                    init_number = int(event_init_path.split('/')[-1].split('.png')[0]) + 1

                    
                    for j in range(10):
                        event_paths.append(os.path.join(event_dir, str(init_number).zfill(4) + '_' + str(j+1) + '.npz'))
                    event_paths = np.array(event_paths)
                    new_info['event_paths'] = event_paths
                    
                    # for k in range(10):
                    #     new_annos.append(infos[i+k]['annos'])
                    # pdb.set_trace()
                    # new_info['annos'] = infos[i]['annos']
                    new_info['seq_annos'] = np.array(infos[i]['annos'])
                    # new_info['eval_annos'] = infos[i]['annos']
                    
                    new_infos.append(new_info)
                infos = new_infos
                
                # pdb.set_trace()
            
            
     
            
            
            waymo_infos.extend(infos)
        
        self.infos.extend(waymo_infos[:])
        
        # self.infos = self.infos[:10]
        
        
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))
        
        # if dist.get_rank() == 0:
        #     pdb.set_trace()
        
        if self.dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[self.mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

    def check_sequence_name_with_all_version(self, seq_file):
        if 's3' in self.root_path:
            if '_with_camera_labels' not in seq_file and\
                not oss_exist(self.data_path, seq_file, self.oss_data_list):
                seq_file = seq_file[:-9] + '_with_camera_labels.tfrecord'
            if '_with_camera_labels' in seq_file and\
                not oss_exist(self.data_path, seq_file, self.oss_data_list):
                seq_file = seq_file.replace('_with_camera_labels', '')
        else:
            if '_with_camera_labels' not in seq_file and not os.path.exists(seq_file):
                seq_file = seq_file[:-9] + '_with_camera_labels.tfrecord'
            if '_with_camera_labels' in seq_file and not os.path.exists(seq_file):
                seq_file = seq_file.replace('_with_camera_labels', '')

        return seq_file

    # def get_infos_and_points(self, idx_list):
    #     infos, points = [], []
    #     for i in idx_list:
    #         lidar_path = self.infos[i]["lidar_path"]
    #         if 's3' in self.root_path:
    #             current_point = np.load(io.BytesIO(self.client.get(lidar_path))) -
    #         else:
    #             current_point = np.load(lidar_path)
    #         infos.append(self.infos[i])
    #         points.append(current_point)

    #     return infos, points
    def get_infos_and_points(self, idx_list):
        infos, points = [], []
        for i in idx_list:
            lidar_path = self.infos[i]["lidar_path"]
            if 's3' in self.root_path:
                lidar_path = str(lidar_path).replace('s3://dataset', 'cluster2:s3://dataset')
                if 'v3' in lidar_path:
                    lidar_path = lidar_path.replace('s3://dataset/waymo/waymo_processed_data_v3', 'cluster2:s3://dataset/waymo/waymo_processed_data_v4')
                current_point = np.load(io.BytesIO(self.client.get(lidar_path))) 
            else:
                # lidar_path = str(lidar_path).replace('/cpfs2/user/matao/workspace/3dal-toolchain-v2/detection/data', '../data')
                
                # if self.mode == 'train':
                # lidar_path = str(lidar_path).replace('/waymo_processed_data_v4', '/processed_data_origin/waymo_processed_data_v4'
                                                    # ).replace('with_camera_labels', 'with_camera_labels/lidar_front')
                
                new_lidar_seq = lidar_path.split('/waymo_processed_data_v4/')[-1]
                lidar_path = os.path.join(self.root_path, 'waymo_processed_data_v4', new_lidar_seq).replace('with_camera_labels', 'with_camera_labels/lidar_fov')
                
                
                # lidar_path = str(lidar_path).replace('with_camera_labels', 'with_camera_labels/lidar_front')
                
                # if self.mode != 'train':
                #     seq_name = lidar_path.split('/')[-1]
                #     seq_idx = int(seq_name.split('.')[0])
                #     if seq_idx % 10 != 0:
                #         new_seq_name = seq_name.replace(str(seq_idx).zfill(4), str(seq_idx // 10 * 10).zfill(4))
                #         lidar_path = lidar_path.replace(seq_name, new_seq_name)
                    
                # print(lidar_path)
                current_point = np.load(lidar_path)
                
                
                # ## lidar sample
                # # if self.mode == 'train':
                # num = current_point.shape[0]
                
                # # num_points = 16192
                # num_points = 65536
                # # pc_points_idx = [][copy.deepcopy(info['annos']) for info in self.infos][copy.deepcopy(info['annos']) for info in self.infos]
                # # pc_sampled = []
                # if num >= num_points:
                #     # pdb.set_trace()
                #     pc_points_idx = np.random.choice(num, num_points, replace=False)
                #     pc_sampled = current_point[pc_points_idx].astype('float32')
                #     current_point = np.array(pc_sampled)
                
                # if self.mode != 'train':
                #     pdb.set_trace()
            
                    
                    # if dist.get_rank() == 0:
                    #     pdb.set_trace()
            
            # if dist.get_rank() == 0:
            #     pdb.set_trace()
            
            infos.append(self.infos[i])
            points.append(current_point)

        return infos, points

    def get_images_and_params(self, current_idx, idx_list):
        imgs_dict = {
            'images': {},
            'extrinsic': {},
            'intrinsic': {},
            'image_shape': {}
        }

        for i in idx_list:
            if not self.load_multi_images:
                if i != current_idx: continue
            # print(len(self.infos))
            # print(idx_list)
            # breakpoint()
            
            img_infos = self.infos[i]['image']
            for key in img_infos.keys():
                if 'path' not in key: continue
                img_path = img_infos[key]
                for j in range(1):
                    img_path = img_path.replace(img_path.split('/')[-2], 'image_%d' % j)
                    if 's3' in self.root_path:
                        image = cv2.imdecode(
                            np.frombuffer(memoryview(self.client.get(img_path.replace('s3://dataset', 'cluster2:s3://dataset'))), np.uint8),
                            cv2.IMREAD_COLOR
                        )
                    else:
                        # if self.mode == 'train':
                        # img_path = str(img_path).replace('/waymo_processed_data_v4', '/processed_data_origin/waymo_processed_data_v4')
                        img_path = str(img_path).replace('/mnt1/LoGoNet/detection/data/waymo/waymo_processed_data_v4', self.data_path)
                        
                        
                        #     seq_name = img_path.split('/')[-1]
                        #     seq_idx = int(seq_name.split('.')[0])
                        #     if seq_idx % 10 != 0:
                        #         new_seq_name = seq_name.replace(str(seq_idx).zfill(4), str(seq_idx // 10 * 10).zfill(4))
                        #         img_path = img_path.replace(seq_name, new_seq_name)

                            # if dist.get_rank() == 0:
                            #     pdb.set_trace()
                        
                        
                        # img_file = cv2.imread(img_path)
                        
                        # if not os.path.exists(img_path):
                        #     print(img_path)
                        
                        # assert img_file.exists()
                        # image = cv2.imread(img_file)
                        image = cv2.imread(img_path)
                        # if self.mode != 'train':
                        #     pdb.set_trace()
                    # normalize images
                    image = image.astype(np.float32)
                    image /= 255.0
                    cam_name = 'camera_%s' % str(j)
                    ### resize image
                    if self.image_scale != 1:
                        new_shape = [int(image.shape[1]*self.image_scale), int(image.shape[0]*self.image_scale)]
                        image = cv2.resize(image, new_shape)
                        img_infos['image_shape_%d' % j] = new_shape[::-1]
                    if cam_name not in imgs_dict['images']:
                        imgs_dict['images'][cam_name] = []
                    imgs_dict['images'][cam_name].append(image)            

        # On waymo dataset, the camera coordinate is not the same
        # with common defination, so we need to swap the axes around
        axes_tf = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]])
        top_lidar_ex = np.array([
            [-0.847772463, -0.530354157, -0.002513657, 1.43],
            [0.530355440, -0.847775367, 0.0001801442, 0.0],
            [-0.002226556, -0.001180410, 0.9999968245, 2.184],
            [0.0, 0.0, 0.0, 1.0]
        ])
        # get the camera related parameters
        for j in range(1):
            cam_name = 'camera_%s' % str(j)
            new_ex_param = np.matmul(axes_tf, np.linalg.inv(img_infos['image_%d_extrinsic' % j]))
            
            imgs_dict['extrinsic'][cam_name] = new_ex_param
            imgs_dict['intrinsic'][cam_name] = img_infos['image_%d_intrinsic' % j]
            imgs_dict['image_shape'][cam_name] = img_infos['image_shape_%d' % j]
        return imgs_dict
    
    def get_events(self, current_idx, idx_list):
        evs_dict = {
            'events': [],
            }

        for i in idx_list:
            if not self.load_multi_images:
                if i != current_idx: continue
            
            event_paths = self.infos[i]['event_paths']
            for i in range(len(event_paths)):
                event_path = event_paths[i]
                voxel = np.load(event_path)['voxel']
                    
                ### resize image
                if self.image_scale != 1:
                    voxel = torch.from_numpy(voxel)
                    voxel = voxel.unsqueeze(0)
                    _, B, H, W = voxel.shape
                    voxel =  F.interpolate(voxel, size=(H//2, W//2))
                    voxel = voxel.squeeze(0).numpy()
                    
                evs_dict['events'].append(voxel)
        return evs_dict
    

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval_detection import WaymoDetectionMetricsEstimator
            eval = WaymoDetectionMetricsEstimator()

            
            
            
            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False),
                fov_flag=self.dataset_cfg.get('EVAL_FOV_FLAG', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        # pdb.set_trace()
        # eval_gt_annos = [copy.deepcopy(info['seq_annos']) for info in self.infos]
        eval_gt_annos = []
        for info in self.infos:
            annos = copy.deepcopy(info['seq_annos'])
            for anno in annos:
                eval_gt_annos.append(anno)
            # eval_gt_annos.append(annos[0])
        
        # eval_det_annos = []
        # for idx, anno in enumerate(det_annos):
        #     if idx % 10 == 0:
        #         eval_det_annos.append(anno)
        #         anno_ = copy.deepcopy(anno)
        #     else:
        #         eval_det_annos.append(anno_)
            
            
        # import pdb; pdb.set_trace()
        # eval_det_annos = eval_det_annos[0::10]
        # eval_gt_annos = eval_gt_annos[0::10]
        ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)

        return ap_result_str, ap_dict


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/cpfs2/shared/public/repos/detection/tools/cfgs/det_dataset_cfgs/waymo_one_sweep.yaml', help='specify the config of dataset')
    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.load(open(args.cfg_file), Loader=yaml.FullLoader))
    class_names=['Vehicle', 'Pedestrian', 'Cyclist']

    file_path ='/cpfs2/shared/public/'
    file_name = 'segment-10868756386479184868_3000_000_3020_000_with_camera_labels'  # training
    # file_name = 'segment-10534368980139017457_4480_000_4500_000_with_camera_labels'  # testing
    info_path = os.path.join(file_path, file_name, '%s.pkl'%file_name)
    with open(info_path, 'rb') as f:
        data_infos = pickle.load(f)


    point_list = []
    point_name = os.listdir(os.path.join(file_path, file_name))
    for name in point_name:
        if len(name) != 8: continue
        point_path = os.path.join(file_path, file_name, name)
        points = np.load(point_path)
        point_list.append(points)

    dataset = WaymoInferenceDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, data_infos=data_infos, point_list=point_list, training=False,
        logger=common_utils.create_logger()
    )
    print("Waymo Inference Dataset contains %d frames." % len(dataset))
    print(dataset[0].keys())
    import pdb; pdb.set_trace()
