# Data Preparation

We provide the dataloaders of Ev-Waymo and DSEC-3DOD together with pre-processed versions of both datasets.

- Ev-Waymo: [Hugging Face](https://huggingface.co/datasets/mickeykang/Ev-Waymo) (one zip per segment, plus `train.txt` / `val.txt`)
- DSEC-3DOD: [Hugging Face](https://huggingface.co/datasets/mickeykang/DSEC-3DOD) (`DSEC.zip`, plus `detection_train_sample_new.txt` / `detection_val_sample_new.txt`) or [Google Drive](https://drive.google.com/drive/folders/1A6XhFxDlqcIgTi28G01fhXBQceaK5vjV?usp=drive_link)

## 1. Directory layout

The dataset configs in `Stage*/detection/tools/cfgs/det_dataset_cfgs/*.yaml` define `DATA_PATH` (relative to the `tools` directory you launch from, or an absolute path) and `PROCESSED_DATA_TAG`. The loaders open `DATA_PATH/PROCESSED_DATA_TAG/<sequence>/...` and read the split files from `DATA_PATH`. With the default values the layout is:

### DSEC-3DOD (`DATA_PATH: ./data`, `PROCESSED_DATA_TAG: dsec_detection`)
```
data
├── detection_train_sample_new.txt
├── detection_val_sample_new.txt
└── dsec_detection
    ├── zurich_city_00_a_00
    │   ├── image_0/        0000.png ...
    │   ├── lidar_fov/      0000.npy ...
    │   ├── disparity/
    │   ├── raw_events/     000208.npy ...               (index = event timestamp index)
    │   ├── voxel/          000208_1.npz ... 000208_10.npz   (generated, see 2.)
    │   ├── zurich_city_00_a_00_fov_bbox_lidar_check.pkl              (Stage 1, 10 FPS labels)
    │   └── zurich_city_00_a_00_interpolate_fov_bbox_lidar_check.pkl  (Stage 2, 100 FPS labels)
    └── ...
```
- `DSEC.zip` extracts to `content/drive/MyDrive/dataset/Ev3DOD/DSEC/zurich_city_*`. Move (or symlink) that `DSEC` folder to `data/dsec_detection`.
- `zurich_city_02_e_01` only contains `raw_events`; it is not part of either split and can be ignored.

### Ev-Waymo (`DATA_PATH: ./data/waymo`, `PROCESSED_DATA_TAG: waymo_processed_data_v4`)
```
data/waymo
├── ImageSets
│   ├── train_sample.txt      (copy of train.txt from Hugging Face)
│   └── val_sample.txt        (copy of val.txt)
└── waymo_processed_data_v4
    ├── segment-xxxx_with_camera_labels
    │   ├── image_0/        0000.png ... 0197.png
    │   ├── lidar_fov/      0000.npy ...
    │   ├── raw_events/     0001.npz ... 0197.npz       (events between frame k-1 and frame k)
    │   ├── voxel/          0001_1.npz ... 0001_10.npz   (generated, see 2.)
    │   ├── segment-xxxx_with_camera_labels_fov_bbox_lidar_check.pkl              (Stage 1)
    │   └── segment-xxxx_with_camera_labels_interpolate_fov_bbox_lidar_check.pkl  (Stage 2)
    └── ...
```
- Keep the folder name `waymo_processed_data_v4`: the info pkl files store absolute paths and the loader strips everything before `/waymo_processed_data_v4/`.

## 2. Generate the event voxel grids
```
cd data
python voxel_generate_dsec.py  --dataset_dir ./dsec_detection
python voxel_generate_waymo.py --dataset_dir ./waymo/waymo_processed_data_v4
```
For every raw event file `<k>` the script writes 10 cumulative sub-window voxel grids `voxel/<k>_1.npz ... <k>_10.npz` (`_j` covers the first `j/10` of the interval). The Stage 2 loaders read `_1` ... `_9` of the interval that follows each 10 FPS keyframe, i.e. the 9 blind-time steps at 100 FPS. The scripts run on the CPU and take a few seconds per file.
