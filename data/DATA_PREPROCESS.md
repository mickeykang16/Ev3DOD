We provide the dataloader of Ev-Waymo and DSEC-3DOD. For ease of use in future research, we release pre-processed versions of the Waymo and DSEC datasets.
Ev-Waymo dataset can be downloaded from the link [here](https://drive.google.com/drive/folders/1Q-7VjcGx_GTrWrgTXmpoEd02qms81QyK?usp=drive_link).
DSEC-3DOD dataset can be downloaded from the link [here](https://drive.google.com/drive/folders/1A6XhFxDlqcIgTi28G01fhXBQceaK5vjV?usp=drive_link).


```
data
├── Ev-Waymo
│   │   │── train.txt
│   │   │── val.txt
│   │   │── segment-xxxxx
│   │   │   ├──raw_events & image_0 & lidar_fov & segment-xxxxx_fov_bbox.pkl & segment-xxxxx_interpolate_fov_bbox.pkl
│   │   │── ...
│   │   │   ├── ...
|
├── DSEC-3DOD
│   │   │── train.txt
│   │   │── val.txt
│   │   │── zurich_city_xxxxx
│   │   │   ├──raw_events & image_0 & lidar_fov & disparity & zurich_city_xxxx_fov_bbox_lidar_check.pkl & zurich_city_xxxx_interpolate_fov_bbox_lidar_check.pkl
│   │   │── ...
│   │   │   ├── ...
|
```

* Generate the event voxel grid by running the following command: 

## Ev-Waymo
```python 
python 
```

