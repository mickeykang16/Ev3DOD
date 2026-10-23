# Getting Started
The dataset configs are located within [tools/cfgs/det_dataset_configs](../detection/tools/cfgs/det_dataset_cfgs), 
and the model configs are located within [tools/cfgs/det_model_configs](../detection/tools/cfgs/det_model_cfgs) for different datasets. 

Pre-trained checkpoint files can be downloaded from [here](https://drive.google.com/drive/folders/1_wl4eHlia9FOdOPKhCjEJ21w9LnG0OuS?usp=drive_link).

Stage 1 is the 10 FPS LiDAR + image detector (no events). It is trained from scratch with the commands below; no extra `--pretrained_model` is needed (the comment about a 1-stage checkpoint inside the configs is inherited from LoGoNet). The resulting checkpoint is passed to Stage 2 with `--pretrained_model`. Prepare the data as described in [data/DATA_PREPROCESS.md](../../data/DATA_PREPROCESS.md) first.


## Training & Testing

### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
```shell script
bash scripts/dist_train_mm.sh ${GPU_IDS} ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --find_unused_parameters
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
bash scripts/dist_train_mm.sh 0,1 2 --cfg_file cfgs/det_model_cfgs/waymo/LoGoNet-1f_play.yaml --extra_tag waymo_stage1 --workers 4 --find_unused_parameters
```

* Testing
``` shell script
bash scripts/dist_test.sh 0,1 2 --ckpt $ckpt_path --cfg_file cfgs/det_model_cfgs/waymo/LoGoNet-1f_play.yaml --workers 0
```

## DSEC-3DOD
* Training
``` shell script
bash scripts/dist_train_mm.sh 0,1 2 --cfg_file cfgs/det_model_cfgs/dsec/LoGoNet-1f_play-pt2.yaml --extra_tag dsec_stage1 --workers 4 --find_unused_parameters
```

* Testing
``` shell script
bash scripts/dist_test.sh 0 1 --ckpt $ckpt_path --cfg_file cfgs/det_model_cfgs/dsec/LoGoNet-1f_play-pt2.yaml --workers 0
```
