# Getting Started
The dataset configs are located within [tools/cfgs/det_dataset_configs](../detection/tools/cfgs/det_dataset_cfgs), 
and the model configs are located within [tools/cfgs/det_model_configs](../detection/tools/cfgs/det_model_cfgs) for different datasets. 

## Training & Testing
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

### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Ev-Waymo

* DSEC-3DOD
