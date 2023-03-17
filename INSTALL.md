### Set up the python environment
* OS: Ubuntu 20.04
* CUDA>=10.1
* We conduct the training on a single NVIDIA A100 GPU
* Clone this repo by `git clone https://github.com/fanegg/UV-Volumes.git`
* Python>=3.8 and PyTorch>=1.7 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended)
    ```
    conda create -n uvvolumes python=3.8
    conda activate uvvolumes

    # make sure that the pytorch cuda is consistent with the system cuda
    conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
    ```
* Install core requirements
    ```
    pip install -r requirements.txt

    cd uvvolumes
    git clone https://github.com/traveller59/spconv --recursive
    cd spconv
    git checkout abf0acf30f5526ea93e687e3f424f62d9cd8313a
    git submodule update --init --recursive
    export CUDA_HOME="/usr/local/cuda-10.1"
    python setup.py bdist_wheel
    cd dist
    pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
    ```

### Set up datasets

#### ZJU-Mocap dataset

1. Download the ZJU-Mocap dataset [here](https://github.com/zju3dv/EasyMocap#zju-mocap).
2. Create a soft link:
    ```
    ROOT=/path/to/uvvolumes
    cd $ROOT/data
    ln -s /path/to/zju_mocap zju_mocap
    ```

#### CMU Panoptic dataset

1. Download the CMU Panoptic dataset [here](http://domedb.perception.cs.cmu.edu/index.html).
2. Process the CMU Panoptic dataset for the binary mask using [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) and for SMPL parameters using [EasyMocap](https://github.com/zju3dv/EasyMocap) (the [tutorial](https://github.com/zju3dv/neuralbody/blob/master/tools/custom) is recommended).
3. Create a soft link:
    ```
    ROOT=/path/to/uvvolumes
    cd $ROOT/data
    ln -s /path/to/cmu_panoptic cmu_panoptic
    ```

#### H36M dataset

1. Download the H36M dataset [here](http://vision.imar.ro/human3.6m/).
2. Process the H36M dataset for the binary mask using [CIHP_PGN](https://github.com/Engineering-Course/CIHP_PGN) and for SMPL parameters using [EasyMocap](https://github.com/zju3dv/EasyMocap).
3. Create a soft link:
    ```
    ROOT=/path/to/uvvolumes
    cd $ROOT/data
    ln -s /path/to/h36m h36m
    ```