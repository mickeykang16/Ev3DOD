import os
import numpy as np
import pdb
import torch
from pathlib import Path
from tqdm import tqdm
import argparse



num_bins = 5
height = 480
width = 640


def events_to_voxel_grid(x, y, p, t, num_bins=num_bins, width=width, height=height):
    t = (t - t[0]).astype('float32')
    t = (t/t[-1])
    x = x.astype('float32')
    y = y.astype('float32')
    pol = p.astype('float32')
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    pol = torch.from_numpy(pol)
    time = torch.from_numpy(t)
    
    with torch.no_grad():
        voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float, requires_grad=False)
        C, H, W = voxel_grid.shape
        t_norm = time
        t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

        x0 = x.int()
        y0 = y.int()
        t0 = t_norm.int()
        if int(pol.min()) == -1: 
            value = pol
        else:
            value = 2*pol-1
        # import pdb; pdb.set_trace()
        for xlim in [x0,x0+1]:
            for ylim in [y0,y0+1]:
                for tlim in [t0,t0+1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < num_bins)
                    interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())
                    index = H * W * tlim.long() + \
                            W * ylim.long() + \
                            xlim.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        mask = torch.nonzero(voxel_grid, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = voxel_grid[mask].mean()
            std = voxel_grid[mask].std()
            if std > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / std
            else:
                voxel_grid[mask] = voxel_grid[mask] - mean
    
    return voxel_grid

def get_event(filepath: Path):
    assert filepath.is_file()
    # x, y, t, p = get_event
    events = np.load(filepath)
    # pdb.set_trace()
    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    
    t = t - t[0]
    # return events
    # x, y, t, p = events[:, 1], events[:, 2], events[:, 0], events[:, 3]
    
    return x, y, t, p


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='.')

    parser.add_argument('--split', type=int, default=0, help='split number')

    args = parser.parse_args()

    # dataset_dir = './stvsr_parse'
    dataset_dir = './dsec_raw_events'
    folder_list_all = os.listdir(dataset_dir)
    folder_list_all.sort()

    total_length = len(folder_list_all)
    if args.split != 0:
        split_len = total_length // 2
        if args.split == 1:
            folder_list_all = folder_list_all[:split_len]
        elif args.split == 2:
            folder_list_all = folder_list_all[split_len:]
            

    
    save_dir = './dsec_voxels_100FPS_from_start'
    
    #filename = 'waymo_val.txt'
    #f = open(filename, 'r')
    #val_lists = f.readlines()
    
    
    for folder_name in folder_list_all:
        print(folder_name)
        
        # if folder_name not in DATA_SPLIT['trainval']:
        #     continue
        seq_dir = os.path.join(dataset_dir, folder_name, 'raw_events')
       
          
        event_vox_save_dir = os.path.join(save_dir, folder_name, 'voxel')
        if not os.path.exists(event_vox_save_dir):
            os.makedirs(event_vox_save_dir)
      
        # event_vox_save_dir_dummy = os.path.join(dataset_dir, folder_name, 'voxel_5bin')
        # import shutil
        # if os.path.exists(event_vox_save_dir_dummy):
        #     shutil.rmtree(event_vox_save_dir_dummy)
        # continue
        
        
        event_path_all = os.listdir(seq_dir)
        event_path_all.sort()
        
        
        # x, y, t, p = self.get_event(Path(self.event_pathstrings[index]))
        
        for ev_path in tqdm(event_path_all):
            event_path = Path(os.path.join(seq_dir, ev_path))
            x, y, t, p = get_event(event_path)
            for i in range(10):
                # time_select = (t.max() + 1)*i*0.1
                time_select = 0
                next_time_select = (t.max() + 1)*(i+1)*0.1
                start_index = (t >= time_select) & (t <= next_time_select)
                
                se_p = p[start_index]
                se_t = t[start_index]
                se_x = x[start_index]
                se_y = y[start_index]
                event_representation = events_to_voxel_grid(se_x, se_y, se_p, se_t)
                np.savez_compressed(os.path.join(event_vox_save_dir, ev_path.replace('.npy', '_' + str(i+1) + '.npz'))
                        , voxel = event_representation)
            
            # np.save(event_vox_save_dir + '/' + str(location) + '/' + disp_gt_pathstrings[index].split('/')[-1].replace('.png', '.npy')
            #         ,event_representation)
    
