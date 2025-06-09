import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
# import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pdb
import os
# import PIL
import PIL.Image
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
import pickle5 as pickle
import glob
import matplotlib.cm as cm
import matplotlib
import cv2
import copy

# import sys
# sys.path.append('./waymo-toolkit')

# breakpoint()
# from waymo_toolkit.viewer import *

def inverse_T(T):
    assert T.shape == (4, 4)
    R = T[:3, :3]
    R_inv = np.linalg.inv(R)
    t = T[:-1, -1]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:-1, -1] = -R_inv @ t
    return T_inv

def get_3d_box_projected_corners(vehicle_to_image, box_to_vehicle):
    """Get the 2D coordinates of the 8 corners of a label's 3D bounding box.
    vehicle_to_image: Transformation matrix from the vehicle frame to the image frame.
    label: The object label
    """

    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)

    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2, 2, 2, 2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k - 0.5), (l - 0.5), (m - 0.5), 1.0])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k, l, m, :] = [v[0] / v[2], v[1] / v[2]]

    return vertices

# # root_path = Path('/home/jaeyoung/data3/dsec/detection')
# root_path = Path('/home/user/jaeyoung/data4/waymo/processed_data_origin/waymo_processed_data_v4')

# # breakpoint()
# save_dir = './test_pred'
# os.makedirs(save_dir, exist_ok=True)

# pickle_file = '../output/det_model_cfgs/waymo/LoGoNet-1f/logo_front_sample_wo_intensity/eval/epoch_10/val/result.pkl'
# with open(pickle_file, 'rb') as f:
#     data = pickle.load(f)
# # pdb.set_trace()

# num_sample = len(data)
# prev_sequence_name = None
# for i in tqdm(range(0, num_sample, 5)):
#     sample = data[i]
    
#     sequence_name = sample['sequence_name']
#     number_str = str(sample['frame_id']).zfill(4)
    
#     pdb.set_trace()
#     if not (prev_sequence_name == sequence_name):
#         gt_pickle_file = root_path / sample['sequence_name'] / (sample['sequence_name'] + '_fov_bbox.pkl')
#         with open(str(gt_pickle_file), 'rb') as f:
#             gt_data = pickle.load(f)
    
#     gt_sample = gt_data[sample['frame_id']]
    
#     img_path = root_path / sample['sequence_name'] / 'image_0' / (number_str + '.png')
#     lidar_path = root_path / sample['sequence_name'] / 'lidar_front' / (number_str + '.npy')
#     img_path = str(img_path)
#     lidar_path = str(lidar_path)
    
#     extr = gt_sample['image']['image_0_extrinsic']
#     intr = gt_sample['image']['image_0_intrinsic']
#     image = cv2.imread(img_path)
#     image = image[...,::-1]
#     lidar = np.load(lidar_path)[:, :3]
    
#     # pdb.set_trace()
#     box_class = sample['name']
#     center = sample['boxes_lidar'][:, :3]
#     dimensions = sample['boxes_lidar'][:, 3:6]
#     heading = sample['boxes_lidar'][:, -1]

#     ones = np.ones((lidar.shape[0],1))

#     # N x 4
#     lidar_homo = np.concatenate((lidar, ones), axis=1)
#     # 4 x N
#     lidar_2_cam = inverse_T(extr) @ lidar_homo.T
#     # lidar_2_cam = lidar_2_cam[:3, :]
#     is_front = lidar_2_cam[2, :] > 0.0
#     projected = intr @ lidar_2_cam
#     projected = projected.T
#     projected = projected[:, :-1] / np.repeat(projected[:, -1:], 2, -1)
#     # breakpoint()
#     H, W, C = image.shape
#     mask = (projected[:, 0] >= 0) & (projected[:, 0] <= W-1) \
#         & (projected[:, 1] >= 0) & (projected[:, 1] <= H-1)
#     mask = is_front & mask
#     projected = projected[mask]
#     projected = np.round(projected).astype(int)
    
#     # image[projected[:, 1], projected[:, 0]] = np.array([255, 0, 0])
    
#     large_dot = [[i, j] for i in range(-1, 2) for j in range(-1, 2)]
    
#     coord = projected.copy()
    
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=30)
#     cmap = cm.jet
#     # m = cm.ScalarMappable(norm=norm, cmap=cmap)
#     color = 255.0 * cmap(norm(lidar_2_cam[2][mask]))[:, :3]
#     color_all = color.copy()

#     for dx, dy in large_dot:
#         if dx == 0 and dy == 0: continue
#         temp = np.zeros_like(projected)
#         # pdb.set_trace()
#         temp[:,0] = np.minimum(np.maximum(0, projected[:, 0] + dx), W-1)
#         temp[:,1] = np.minimum(np.maximum(0, projected[:, 1] + dy), H-1)
#         temp = temp.astype(int)
#         coord = np.concatenate([coord, temp], axis = 0)
#         color_all = np.concatenate([color_all, color], axis = 0)
        
#     # breakpoint()
    
#     # image[coord[:, 1], coord[:, 0]] = np.array([255, 0, 0])
#     image[coord[:, 1], coord[:, 0]] = color_all
#     im = PIL.Image.fromarray(image)
    
#     save_path = Path(save_dir) / sequence_name / (number_str+'.png')
#     # pdb.set_trace()
#     # breakpoint()
    
#     # os.makedirs(str(Path(save_dir) / sequence_name), exist_ok=True)
#     # im.save(str(save_path))
    
#     for cls, dim, cent, yaw in zip(box_class, dimensions, center, heading):
#         if cls == 'Sign': continue
#         elif cls == 'Vehicle': color = 'red'
#         elif cls == 'Pedestrian': color = 'green'
#         elif cls == 'Cyclist': color = 'blue'
#         # print(cls)
        
        
#         if cent[0] < 0.0:continue
#         # point1 = 
#         tx, ty, tz = cent[0], cent[1], cent[2]
#         c = np.cos(yaw)
#         s = np.sin(yaw)
#         # dim -> l, w, h
#         sl, sw, sh = dim[0], dim[1], dim[2]
#         box_to_vehicle = np.array(
#             [[sl * c, -sw * s, 0, tx], [sl * s, sw * c, 0, ty], [0, 0, sh, tz], [0, 0, 0, 1]]
#         )
        
#         # breakpoint()
#         draw = ImageDraw.Draw(im)
#         vehicle_to_image = intr @ inverse_T(extr)
#         vertices = get_3d_box_projected_corners(vehicle_to_image, box_to_vehicle)
#         if vertices is None: continue
#         for k in [0, 1]:
#             for l in [0, 1]:
#                 for idx1, idx2 in [
#                     ((0, k, l), (1, k, l)),
#                     ((k, 0, l), (k, 1, l)),
#                     ((k, l, 0), (k, l, 1)),
#                 ]:
#                     draw.line((vertices[idx1][0], vertices[idx1][1], vertices[idx2][0], vertices[idx2][1]), fill=color, width=3)
#     os.makedirs(str(Path(save_dir) / sequence_name), exist_ok=True)      
#     im.save(str(save_path))
    
#     prev_sequence_name = sequence_name

def visualize_event(event_array):
    
    sensor_size = event_array.shape
    
    vis_tensor_ = np.zeros((sensor_size[1], sensor_size[2], 3), np.uint8)
    
    event_sum = event_array.sum(0)
    mask = event_sum >0
    neg_mask = event_sum <0

    vis_tensor_[mask, 0] = 255
    vis_tensor_[mask, 1] = 0
    vis_tensor_[mask, 2] = 0
    vis_tensor_[neg_mask, 0] = 0
    vis_tensor_[neg_mask, 1] = 0
    vis_tensor_[neg_mask, 2] = 255
    return vis_tensor_

    

            
def save_data(data: dict):
    pcd = data['pcd'] # N x 3
    image = data['image']
    image = image.transpose([1, 2, 0]) * 255.0
    image_orig = image[:, :, [2,1,0]]
    extr = data['extr']
    intr = data['intr']
    events = data['events']
    gt_boxes = data['gt_boxes']
    aug_matrix_inv = data['aug_mat']
    
    if aug_matrix_inv is not None:
        for aug_type in ['translate', 'rescale', 'rotate', 'flip']:
            if aug_type in aug_matrix_inv:
                if aug_type == 'translate':
                    pcd = pcd + np.array(aug_matrix_inv[aug_type])
                else:
                    pcd = pcd @ np.array(aug_matrix_inv[aug_type])
    save_dir = Path('../output/images')
    
    pdb.set_trace()
    for f in range(len(gt_boxes)):
        lidar = copy.deepcopy(pcd)
        image = copy.deepcopy(image_orig).astype(np.uint8)
        if f == 0:
            event = np.zeros_like(events[0])
        else:
            event = events[f-1]
        event_tensor = visualize_event(event)
        image = cv2.addWeighted(image, 1.0, event_tensor, 1.0, 0)
        
        gt_box = gt_boxes[f]
        center = gt_box[:, :3]
        dimensions = gt_box[:, 3:6]
        heading = gt_box[:, 6]
        
        
        ones = np.ones((lidar.shape[0],1))
        # pdb.set_trace()
        # N x 4
        lidar_homo = np.concatenate((lidar, ones), axis=1)
        # 4 x N
        lidar_2_cam = extr @ lidar_homo.T
        # lidar_2_cam = lidar_2_cam[:3, :]
        is_front = lidar_2_cam[2, :] > 0.0
        projected = intr @ lidar_2_cam
        projected = projected.T
        projected = projected[:, :-1] / np.repeat(projected[:, -1:], 2, -1)
        # breakpoint()
        H, W, C = image.shape
        mask = (projected[:, 0] >= 0) & (projected[:, 0] <= W-1) \
            & (projected[:, 1] >= 0) & (projected[:, 1] <= H-1)
        mask = is_front & mask
        projected = projected[mask]
        projected = np.round(projected).astype(int)
        
        # image[projected[:, 1], projected[:, 0]] = np.array([255, 0, 0])
        
        large_dot = [[i, j] for i in range(-1, 2) for j in range(-1, 2)]
        
        coord = projected.copy()
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=30)
        cmap = cm.jet
        color = 255.0 * cmap(norm(lidar_2_cam[2][mask]))[:, :3]
        color_all = color.copy()

        for dx, dy in large_dot:
            if dx == 0 and dy == 0: continue
            temp = np.zeros_like(projected)
            temp[:,0] = np.minimum(np.maximum(0, projected[:, 0] + dx), W-1)
            temp[:,1] = np.minimum(np.maximum(0, projected[:, 1] + dy), H-1)
            temp = temp.astype(int)
            coord = np.concatenate([coord, temp], axis = 0)
            color_all = np.concatenate([color_all, color], axis = 0)
            
        
        # pdb.set_trace()
        image[coord[:, 1], coord[:, 0]] = color_all
        im = PIL.Image.fromarray(image)
        
        for dim, cent, yaw in zip(dimensions, center, heading):
            color = 'red'
            
            if cent[0] < 0.0:continue
            # point1 = 
            tx, ty, tz = cent[0], cent[1], cent[2]
            c = np.cos(yaw)
            s = np.sin(yaw)
            # dim -> l, w, h
            sl, sw, sh = dim[0], dim[1], dim[2]
            box_to_vehicle = np.array(
                [[sl * c, -sw * s, 0, tx], [sl * s, sw * c, 0, ty], [0, 0, sh, tz], [0, 0, 0, 1]]
            )

            draw = ImageDraw.Draw(im)
            vehicle_to_image = intr @ extr
            vertices = get_3d_box_projected_corners(vehicle_to_image, box_to_vehicle)
            if vertices is None: continue
            for k in [0, 1]:
                for l in [0, 1]:
                    for idx1, idx2 in [
                        ((0, k, l), (1, k, l)),
                        ((k, 0, l), (k, 1, l)),
                        ((k, l, 0), (k, l, 1)),
                    ]:
                        pass
                        draw.line((vertices[idx1][0], vertices[idx1][1], vertices[idx2][0], vertices[idx2][1]), fill=color, width=3)
        
        save_name = save_dir / f'sample_{f}.png'
        os.makedirs(str(Path(save_dir)), exist_ok=True)
        im.save(str(save_name))
        
    return
    