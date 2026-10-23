# Getting Started
The dataset configs are located within [tools/cfgs/det_dataset_configs](../detection/tools/cfgs/det_dataset_cfgs), 
and the model configs are located within [tools/cfgs/det_model_configs](../detection/tools/cfgs/det_model_cfgs) for different datasets. 

Pre-trained checkpoint files can be downloaded from [here](https://drive.google.com/drive/folders/1_wl4eHlia9FOdOPKhCjEJ21w9LnG0OuS?usp=drive_link).

## Pipeline
Ev-3DOD is trained in two stages, and Stage 2 needs a Stage 1 checkpoint:

1. **Stage 1** (`Stage1/`, 10 FPS, LiDAR + image, no events): train from scratch with `Stage1/detection/tools/scripts/dist_train_mm.sh`, or use the released `*_stage1.pth`.
2. **Stage 2** (`Stage2/`, 100 FPS with events): start from the Stage 1 checkpoint with `--pretrained_model`. Stage 2 freezes the modules inherited from Stage 1 (3D/2D backbones, dense head, image backbone, LiDAR-image RoI head; `requires_grad=False` and kept in `eval()`), and trains the event backbone, the event-voxel fusion and the event RoI head (`ROI_HEAD_EV`).
3. **Evaluation** at 100 FPS with `scripts/dist_test.sh` and the Stage 2 checkpoint.

`--pretrained_model` copies every tensor whose name and shape match into the Stage 2 model; `--ckpt` is only for resuming an interrupted Stage 2 run. Prepare the data as described in [data/DATA_PREPROCESS.md](../../data/DATA_PREPROCESS.md) first (the event voxel grids are required for Stage 2).


## Training & Testing

### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
```shell script
bash scripts/dist_train_mm.sh ${GPU_IDS} ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --pretrained_model ${STAGE1_CKPT} --find_unused_parameters
```
(`dist_train.sh` / `train.py` are LiDAR-only leftovers from LoGoNet; always use `dist_train_mm.sh` / `train_mm.py`.)

### Test and evaluate the pretrained models

* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

## Ev-Waymo 
* Training
``` shell script
bash scripts/dist_train_mm.sh 0 1 --cfg_file cfgs/det_model_cfgs/waymo/LoGoNet-1f-small-only-position-cls-fg0.75-bg0.25.yaml --extra_tag waymo_stage2 --find_unused_parameters --pretrained_model $pretrained_waymo_stage1_path
```

* Testing
``` shell script
bash scripts/dist_test.sh 0,1 2 --ckpt $checkpoint_path --cfg_file cfgs/det_model_cfgs/waymo/LoGoNet-1f-small-only-position-cls-fg0.75-bg0.25.yaml --workers 0
```

## DSEC-3DOD
* Training
``` shell script
bash scripts/dist_train_mm.sh 0 1 --cfg_file cfgs/det_model_cfgs/dsec/LoGoNet-1f-small-only-position-image0.5-event0.5-reg1.0.yaml --extra_tag dsec_stage2 --find_unused_parameters --pretrained_model $pretrained_dsec_stage1_path
```

* Testing
``` shell script
bash scripts/dist_test.sh 0,1 2 --ckpt $checkpoint_path --cfg_file  cfgs/det_model_cfgs/dsec/LoGoNet-1f-small-only-position-image0.5-event0.5-reg1.0.yaml --workers 0
```
