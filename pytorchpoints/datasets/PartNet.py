import os
import copy
import numpy as np
import pickle
import h5py
import torch
import torchvision
from pytorchpoints.functional import pc_normalize, get_part_seg_features, dict_as_tensor, grid_subsampling
from pytorchpoints.config import configurable
from pytorchpoints.config import kfg
from .build import DATASETS_REGISTRY
from .transforms import PointcloudScaleAndJitter, PointcloudToTensor, PointcloudRandomRotate

__all__ = ["PartNetSeg"]


def get_label_to_names():
    label_to_names = { 0: 'Bed',           1: 'Bottle',            2: 'Chair',  3: 'Clock',
                       4: 'Dishwasher',    5: 'Display',           6: 'Door',   7: 'Earphone',
                       8: 'Faucet',        9: 'Knife',            10: 'Lamp',  11: 'Microwave',
                      12: 'Refrigerator', 13: 'StorageFurniture', 14: 'Table', 15: 'TrashCan',
                      16: 'Vase'}
    return label_to_names


@DATASETS_REGISTRY.register()
class PartNetSeg:
    @configurable
    def __init__(
        self,
        stage: str,
        data_root: str,
        infeats_dim: int,
        num_parts,
        transforms,
    ):
        self.stage = stage
        self.data_root = data_root
        self.infeats_dim = infeats_dim
        self.transforms = transforms
        self.label_to_names = get_label_to_names()
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.num_parts = num_parts
        self.shuffle_points = True if (stage == 'train') else False
        self.folder = 'PartNet'
        self.data_dir = os.path.join(self.data_root, self.folder, 'sem_seg_h5')

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = {
            "stage": stage,
            "data_root": cfg.DATALOADER.DATA_ROOT,
            "infeats_dim": cfg.MODEL.INFEATS_DIM,
            "num_parts": cfg.DATALOADER.NUM_PARTS,
        }

        if stage == "train":
            transforms = torchvision.transforms.Compose([
                PointcloudToTensor(),
                PointcloudScaleAndJitter(scale_low=cfg.AUG.SCALE_LOW, scale_high=cfg.AUG.SCALE_HIGH,
                    std=cfg.AUG.NOISE_STD, clip=cfg.AUG.NOISE_CLIP),
                PointcloudRandomRotate(cfg.AUG.X_ANGLE_RANGE, cfg.AUG.Y_ANGLE_RANGE, cfg.AUG.Z_ANGLE_RANGE)])
        else:
            transforms = torchvision.transforms.Compose([PointcloudToTensor()])
        ret.update({ "transforms": transforms })
        return ret

    def _load_seg(self, filelist):
        points = []
        labels_seg = []
        folder = os.path.dirname(filelist)
        for line in open(filelist):
            data = h5py.File(os.path.join(folder, line.strip()), mode='r')
            points.append(data['data'][...].astype(np.float32))
            labels_seg.append(data['label_seg'][...].astype(np.int32))

        return (np.concatenate(points, axis=0),
                np.concatenate(labels_seg, axis=0))

    def load_data(self, cfg):
        datalist = []
        filename = os.path.join(self.data_root, self.folder, '{}_data.pkl'.format(self.stage))
        if not os.path.exists(filename):
            print(f"Preparing PartNetSeg data")
            for class_id, class_name in self.label_to_names.items():
                split_filelist = os.path.join(self.data_dir, '{}-{}'.format(class_name, 3), '{:s}_files.txt'.format(self.stage))
                split_points, split_labels = self._load_seg(split_filelist)
                N = split_points.shape[0]
                for i in range(N):
                    pc = split_points[i]
                    pc = pc_normalize(pc)
                    pc = pc[:, [0, 2, 1]]
                    pcl = split_labels[i]
                    datalist.append({
                        kfg.POINTS: pc,
                        kfg.POINTS_LABELS: pcl,
                        kfg.LABELS: np.array(class_id),
                    })

            with open(filename, 'wb') as f:
                pickle.dump(datalist, f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                datalist = pickle.load(f)
                print(f"{filename} loaded successfully")
        self.num_points = datalist[0][kfg.POINTS].shape[0]
        return datalist

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        current_points = dataset_dict[kfg.POINTS]
        current_points_labels = dataset_dict[kfg.POINTS_LABELS]
        if self.shuffle_points:
            shuffle_choice = np.random.permutation(np.arange(self.num_points))
            current_points = current_points[shuffle_choice, :]
            current_points_labels = current_points_labels[shuffle_choice]

        if self.transforms is not None:
            current_points = self.transforms(current_points)
        features = get_part_seg_features(self.infeats_dim, current_points)

        ret = {
            kfg.POINTS: current_points,
            kfg.FEATS: features,      
        }
        dict_as_tensor(ret)
        ret.update({
            kfg.POINTS_LABELS: torch.from_numpy(current_points_labels).type(torch.int64),
            kfg.MASKS: torch.ones(self.num_points).type(torch.int32),
            kfg.LABELS: torch.from_numpy(dataset_dict[kfg.LABELS]).type(torch.int64),
        })

        return ret