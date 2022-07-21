import os
import copy
import numpy as np
import pickle
import torch
import torchvision
from pytorchpoints.functional import pc_normalize, get_cls_features, dict_as_tensor, grid_subsampling
from pytorchpoints.config import configurable
from pytorchpoints.config import kfg
from .build import DATASETS_REGISTRY
from .transforms import PointcloudScaleAndJitter, PointcloudToTensor, PointcloudRandomRotate

__all__ = ["ModelNet40Cls"]


def get_label_to_names():
    label_to_names = {  0: 'airplane',    1: 'bathtub',    2: 'bed',      3: 'bench',        4: 'bookshelf',
                        5: 'bottle',      6: 'bowl',       7: 'car',      8: 'chair',        9: 'cone',
                       10: 'cup',        11: 'curtain',   12: 'desk',    13: 'door',        14: 'dresser',
                       15: 'flower_pot', 16: 'glass_box', 17: 'guitar',  18: 'keyboard',    19: 'lamp',       
                       20: 'laptop',     21: 'mantel',    22: 'monitor', 23: 'night_stand', 24: 'person',
                       25: 'piano',      26: 'plant',     27: 'radio',   28: 'range_hood',  29: 'sink',
                       30: 'sofa',       31: 'stairs',    32: 'stool',   33: 'table',       34: 'tent',
                       35: 'toilet',     36: 'tv_stand',  37: 'vase',    38: 'wardrobe',    39: 'xbox' }
    return label_to_names


@DATASETS_REGISTRY.register()
class ModelNet40Cls:
    @configurable
    def __init__(
        self,
        stage: str,
        data_root: str,
        num_points: int,
        infeats_dim: int,
        subsampling: float,
        batch_aug: bool,
        transforms,
    ):
        self.stage = stage
        self.data_root = data_root
        self.num_points = num_points
        self.subsampling = subsampling
        self.batch_aug = batch_aug
        self.infeats_dim = infeats_dim
        self.transforms = transforms
        self.use_normal = (infeats_dim >= 6)
        self.label_to_names = get_label_to_names()
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.folder = 'ModelNet40'
        self.data_dir = os.path.join(self.data_root, self.folder, 'modelnet40_normal_resampled')
 
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = {
            "stage": stage,
            "data_root": cfg.DATALOADER.DATA_ROOT,
            "num_points": cfg.DATALOADER.NUM_POINTS,
            "infeats_dim": cfg.MODEL.INFEATS_DIM,
            "subsampling": cfg.DATALOADER.SUBSAMPLING,
            "batch_aug": cfg.AUG.BATCH_AUG
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

    def load_data(self, cfg):
        datalist = []
        if self.stage == 'train':
            names = np.loadtxt(os.path.join(self.data_dir, 'modelnet40_train.txt'), dtype=np.str)
        elif self.stage == 'test':
            names = np.loadtxt(os.path.join(self.data_dir, 'modelnet40_test.txt'), dtype=np.str)
        else:
            raise KeyError(f"ModelNet40 has't split: {self.stage}")    

        if self.num_points == 1024:
            if self.stage == 'train':
                np_data = 2048
            elif self.stage == 'test':
                np_data = 1024
            else:
                raise KeyError(f"ModelNet40 has't split: {self.stage}")
            filename = os.path.join(self.data_root, self.folder,
                                    '{}_{}_data.pkl'.format(self.stage, np_data))
        else:
            filename = os.path.join(self.data_root, self.folder, '{}_{:.3f}_data.pkl'.format(self.stage, self.subsampling))
        if not os.path.exists(filename):
            print(f"Preparing ModelNet40 data with subsampling_parameter={self.subsampling}")
            if self.num_points == 1024:
                raise KeyError(f"No ModelNet40 1024p data")
            # Collect point clouds
            for i, cloud_name in enumerate(names):
                # Read points
                class_folder = '_'.join(cloud_name.split('_')[:-1])
                txt_file = os.path.join(self.data_dir, class_folder, cloud_name) + '.txt'
                data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)
                pc = data[:, :3]
                pc = pc_normalize(pc)
                normal = data[:, 3:]
                label = np.array(self.name_to_label[class_folder])
                # Subsample
                if self.subsampling > 0:
                    pc, normal = grid_subsampling(pc, features=normal, sampleDl=self.subsampling)

                datalist.append({
                    kfg.POINTS: pc,
                    kfg.NORMALS: normal,
                    kfg.LABELS: label,
                })

            with open(filename, 'wb') as f:
                pickle.dump(datalist, f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                datalist = pickle.load(f)
                print(f"{filename} loaded successfully")
        print(f"{self.stage} dataset has {len(datalist)} data with  {self.num_points} points")
        return datalist

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        current_points = dataset_dict[kfg.POINTS][:, [0, 2, 1]]
        cur_num_points = current_points.shape[0]
        if cur_num_points >= self.num_points:
            choice = np.random.choice(cur_num_points, self.num_points)
            current_points = current_points[choice, :]
            if self.use_normal:
                current_normals = dataset_dict[kfg.NORMALS]
                current_normals = current_normals[choice, :]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            padding_num = self.num_points - cur_num_points
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            padding_choice = np.random.choice(cur_num_points, padding_num)
            choice = np.hstack([shuffle_choice, padding_choice])
            current_points = current_points[choice, :]
            if self.use_normal:
                current_normals = dataset_dict[kfg.NORMALS]
                current_normals = current_normals[choice, :]
            mask = torch.cat([torch.ones(cur_num_points), torch.zeros(padding_num)]).type(torch.int32)

        label = torch.from_numpy(dataset_dict[kfg.LABELS]).type(torch.int64)

        if self.use_normal:
            current_points = np.hstack([current_points, current_normals])

        if self.transforms is not None:
            current_points_aug = self.transforms(current_points)
        else:
            current_points_aug = current_points
        pc = current_points_aug[:, :3]
        normal = current_points_aug[:, 3:]
        features = get_cls_features(self.infeats_dim, pc, normal)
  
        ret = {
            kfg.POINTS: pc,
            kfg.MASKS: mask,
            kfg.FEATS: features,
            kfg.LABELS: label,
        }
        dict_as_tensor(ret)

        if self.stage == 'train' and self.batch_aug:
            assert self.transforms is not None
            current_points_aug2 = self.transforms(current_points)

            pc2 = current_points_aug2[:, :3]
            normal2 = current_points_aug2[:, 3:]
            features2 = get_cls_features(self.infeats_dim, pc2, normal2)

            ret2 = {
                kfg.POINTS: pc2,
                kfg.MASKS: mask,
                kfg.FEATS: features2,
                kfg.LABELS: label,
            }
            dict_as_tensor(ret2)

            return {'aug1': ret, 'aug2': ret2}
        return ret

