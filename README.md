# SPE-Net: Boosting Point Cloud Analysis via Rotation Robustness Enhancement
This repository is for **SPE-Net: Boosting Point Cloud Analysis via Rotation Robustness Enhancement**. The repository is based on the open-source codebase [Detectron2](https://github.com/facebookresearch/detectron2.git) and [CloserLook3D](https://github.com/zeliu98/CloserLook3D.git).

## Requiremenets
* Linux with Python ≥ 3.6
* PyTorch ≥ 1.8
* fvcore
* pycocotools
* Java 1.8.0

## Installation
- Prepare the target dataset following the instructions from the codebase CloserLook3D, and put the pre-processed data in the folder `./dataset`.
- Run pytorchpoints/init.sh to compile the C++ code.

## Train SPE-Net Model
configs/spe_net_modelnet40_so3.yaml is the config file for training SPE-Net model on ModelNet40. Run script `python3 train_net.py --num-gpus 4 --config-file configs/spe_net_modelnet40_so3.yaml`.

pytorchpoints/modeling/backbones/resnet.py includes the implementation for **SPE-Net** overall architecture.

pytorchpoints/modeling/local_aggregation/spe_mlp.py includes the implementation for **SPE-MLP**.

## Acknowledgements
Thanks the contribution of [Detectron2](https://github.com/facebookresearch/detectron2.git), [CloserLook3D](https://github.com/zeliu98/CloserLook3D.git) and the PyTorch team.
