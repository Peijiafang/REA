# Introduction
This is a Pytorch implementation of "Re-Attention Is All You Need: Memory-Efficient Scene Text Detection
via Re-Attention on Uncertain Regions
". 

Most of the code is inherited from [DB](https://github.com/MhLiao/DB).

## Installation

The installation step is same as DB.

### Requirements:
- Python3
- PyTorch >= 1.2 
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (10.1 is recommended)


```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name REA -y
  conda activate REA

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install -r requirement.txt

  # install PyTorch with cuda-10.1
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

  # build deformable convolution opertor
  # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
  # make sure GCC >= 4.9
  # you need to delete the build directory before you re-build it.
  echo $CUDA_HOME
  cd assets/ops/dcn/
  python setup.py build_ext --inplace

```

## Models

### Models from ours
```
  model/DB    -- downloaded from DB, used as the SynthText pretrained weight, and evaluate DB model
  model/synth_deform_resnet50_BiFPN_1layer    -- used to finetune our models, is trained from SynthText
  model/synth_deform_resnet50_SegDetector    -- used to finetune other papers, is trained from SynthText
  model/baseline/experiment    -- the config files for the four datasets
  model/baseline/*_visualize    -- the visualization of our models
  model/baseline/{dataset}*BiFPN*    -- the model of the datasets, trained from our methods
  model/baseline/{dataset}*SegDetector*    -- the model of the datasets, trained from DB method
```

### Models from DB
Download Trained models [Baidu Drive](https://pan.baidu.com/s/1vxcdpOswTK6MxJyPIJlBkA) (download code: p6u3), [Google Drive](https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG).
```
  pre-trained-model-synthtext   -- used to finetune models, not for evaluation
  td500_resnet18
  td500_resnet50
  totaltext_resnet18
  totaltext_resnet50
```

## Datasets
The root of the dataset directory can be ```REA/datasets/```.

An example of the path of ctw1500:
```
  datasets
  └ ctw1500
    ├ train_images      -- file named as xxxx.jpg
    ├ train_gts         -- file named as xxxx.jpg.txt
    ├ train_list.txt    -- the list of files in train_images
    ├ test_images       -- file named as xxxx.jpg
    ├ test_gts          -- file named as xxxx.jpg.txt
    └ test_list.txt     -- the list of files in test_images
```

## Testing
### Prepar dataset
An example of the path of test images: 
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```
The data root directory and the data list file can be defined in ```experiments/{dataset}/base_{dataset}.yaml```

### Demo

Run the model inference with a single image. Here is an example:

```
python demo.py experiments/total_text/totaltext_deform_resnet50_bifpn_1layer_zoom_local.yaml --image_path datasets/total_text/test_images/img10.jpg --resume model/totaltext_deform_resnet50_BiFPN_1layer_select_blur --polygon --box_thresh 0.7 --visualize
```

The results can be find in `demo_results`.
### Evaluate the performance
```
python eval.py experiments/total_text/totaltext_deform_resnet50_bifpn_1layer_zoom_local.yaml --resume model/totaltext_deform_resnet50_BiFPN_1layer_select_blur --polygon --box_thresh 0.5

python eval.py experiments/td500/td500_deform_resnet50_bifpn_1layer_zoom_local.yaml --resume model/td500_deform_resnet50_BiFPN_1layer_select_blur --box_thresh 0.5

python eval.py experiments/ic15/ic15_deform_resnet50_bifpn_1layer_zoom_local.yaml --resume model/ic15_deform_resnet50_BiFPN_1layer_select_blur --box_thresh 0.6

python eval.py experiments/ctw1500/ctw_deform_resnet50_BiFPN_1layer_select_blur.yaml --resume model/ctw_deform_resnet50_BiFPN_1layer_select_blur --polygon --box_thresh 0.6
```

The results should be as follows:

|   Model    | precision | recall | F-measure |
| :--------: | :-------: | :----: | :-------: |
|    IC15    |   **82.7**    |  79.2  |   **80.9**    |
|   TD500    |   **92.8**    |  **84.2**  |   **88.3**    |
|  CTW1500   |   83.0    |  76.0  |   **79.3**    |
| Total-Text |   **86.1**    |  **80.7**  |   **83.3**    |


```box_thresh``` can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. ```polygon``` is only used for arbitrary-shape text dataset. The size of the input images are defined in ```validate_data->processes->AugmentDetectionData``` in ```base_*.yaml```.

## Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.

- from scratch
  - ```python train.py path-to-yaml-file```

- from pretrained
  - ```python train.py path-to-yaml-file --resume pretrained-model```


### Modules
- BiFPN + DB
  - ```decoders/seg_detector.py``` 
- Zooming in module + integrating module
  - ```structure/model.py```
