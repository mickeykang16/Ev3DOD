# Getting Started
The dataset configs are located within [tools/cfgs/det_dataset_configs](../detection/tools/cfgs/det_dataset_cfgs), 
and the model configs are located within [tools/cfgs/det_model_configs](../detection/tools/cfgs/det_model_cfgs) for different datasets. 

Pre-trained checkpoint files can be downloaded from [here](https://drive.google.com/drive/folders/1_wl4eHlia9FOdOPKhCjEJ21w9LnG0OuS?usp=drive_link).


## Training & Testing

### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

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
