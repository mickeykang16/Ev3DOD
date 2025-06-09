## Installation
- install the virtual environment and pytorch:
  ```
  conda create --name ev3d_stage2 python=3.6
  source activate env_name
  pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

- install cmake: `conda install cmake`

- install sparse conv: `pip install spconv-cu111`

- install Waymo evaluation module: `pip install waymo-open-dataset-tf-2-0-0`

- install the requirements of LoGoNet: `pip install -r requirements.txt`

### The following procedures can be performed via bash commands
```
bash setup_py.sh
```

- install the requirements of image_modules: `cd detection/models/image_modules/swin_model && pip install -r requirements.txt && python setup.py develop`

- compile LoGoNet:
  ```
  cd utils && python setup.py develop
  ```
- compile the specific algorithm module:
  ```
  cd detection  && python setup.py develop
  ```
- compile the specific dcn module:
  ```
  cd detection/al3d_det/models/ops  && python setup.py develop
  ```