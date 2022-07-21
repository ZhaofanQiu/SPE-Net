import os
import copy
import numpy as np
import pickle
import json
import torch
import torchvision
from pytorchpoints.functional import pc_normalize, get_part_seg_features, dict_as_tensor, grid_subsampling
from pytorchpoints.config import configurable
from pytorchpoints.config import kfg
from .build import DATASETS_REGISTRY
from .transforms import PointcloudScaleAndJitter, PointcloudToTensor, PointcloudRandomRotate

__all__ = ["ShapeNetPartSeg"]


def get_label_to_names():
    label_to_names = {0: 'Airplane',    1: 'Bag',       2: 'Cap',           3: 'Car',
                      4: 'Chair',       5: 'Earphone',  6: 'Guitar',        7: 'Knife',
                      8: 'Lamp',        9: 'Laptop',    10: 'Motorbike',    11: 'Mug',
                      12: 'Pistol',     13: 'Rocket',   14: 'Skateboard',   15: 'Table'}
    return label_to_names


@DATASETS_REGISTRY.register()
class ShapeNetPartSeg:
    @configurable
    def __init__(
        self,
        stage: str,
        data_root: str,
        num_points: int,
        infeats_dim: int,
        num_parts,
        transforms,
    ):
        self.stage = stage
        self.data_root = data_root
        self.num_points = num_points
        self.infeats_dim = infeats_dim
        self.transforms = transforms
        self.label_to_names = get_label_to_names()
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.num_parts = num_parts
        self.shuffle_points = True if (stage == 'train') else False
        self.folder = 'ShapeNetPart'
        self.data_dir = os.path.join(self.data_root, self.folder, 'shapenetcore_partanno_segmentation_benchmark_v0')

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = {
            "stage": stage,
            "data_root": cfg.DATALOADER.DATA_ROOT,
            "num_points": cfg.DATALOADER.NUM_POINTS,
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

    def load_data(self, cfg):
        self.category_and_synsetoffset = [['Airplane', '02691156'],
                                          ['Bag', '02773838'],
                                          ['Cap', '02954340'],
                                          ['Car', '02958343'],
                                          ['Chair', '03001627'],
                                          ['Earphone', '03261776'],
                                          ['Guitar', '03467517'],
                                          ['Knife', '03624134'],
                                          ['Lamp', '03636649'],
                                          ['Laptop', '03642806'],
                                          ['Motorbike', '03790512'],
                                          ['Mug', '03797390'],
                                          ['Pistol', '03948459'],
                                          ['Rocket', '04099429'],
                                          ['Skateboard', '04225987'],
                                          ['Table', '04379243']]
        synsetoffset_to_category = {s: n for n, s in self.category_and_synsetoffset}

        # Train split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_train_file_list.json')
        with open(split_file, 'r') as f:
            train_files = json.load(f)
        train_files = [name[11:] for name in train_files]

        # Val split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_val_file_list.json')
        with open(split_file, 'r') as f:
            val_files = json.load(f)
        val_files = [name[11:] for name in val_files]

        # Test split
        split_file = os.path.join(self.data_dir, 'train_test_split', 'shuffled_test_file_list.json')
        with open(split_file, 'r') as f:
            test_files = json.load(f)
        test_files = [name[11:] for name in test_files]

        split_files = {'train': train_files,
                       'trainval': train_files + val_files,
                       'val': val_files,
                       'test': test_files
                       }
        files = split_files[self.stage]

        datalist = []
        filename = os.path.join(self.data_root, self.folder, '{}_data.pkl'.format(self.stage))
        if not os.path.exists(filename):
            print(f"Preparing ShapeNetPartSeg data")
            for i, file in enumerate(files):
                # Get class
                synset = file.split('/')[0]
                class_name = synsetoffset_to_category[synset]
                cls = self.name_to_label[class_name]
                cls = np.array(cls)
                # Get filename
                file_name = file.split('/')[1]
                # Load points and labels
                pc = np.loadtxt(os.path.join(self.data_dir, synset, 'points', file_name + '.pts')).astype(
                    np.float32)
                pc = pc_normalize(pc)
                pc = pc[:, [0, 2, 1]]
                pcl = np.loadtxt(os.path.join(self.data_dir, synset, 'points_label', file_name + '.seg')).astype(
                    np.int64) - 1

                datalist.append({
                    kfg.POINTS: pc,
                    kfg.POINTS_LABELS: pcl,
                    kfg.LABELS: np.array(cls),
                })
            with open(filename, 'wb') as f:
                pickle.dump(datalist, f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                datalist = pickle.load(f)
                print(f"{filename} loaded successfully")

        return datalist

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        current_points = dataset_dict[kfg.POINTS]
        current_points_labels = dataset_dict[kfg.POINTS_LABELS]
        cur_num_points = current_points.shape[0]
        if cur_num_points >= self.num_points:
            if self.shuffle_points:
                choice = np.random.choice(cur_num_points, self.num_points)
            else:
                choice = np.arange(stop=self.num_points)
            current_points = current_points[choice, :]
            current_points_labels = current_points_labels[choice]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            padding_num = self.num_points - cur_num_points
            if self.shuffle_points:
                shuffle_choice = np.random.permutation(np.arange(cur_num_points))
                padding_choice = np.random.choice(cur_num_points, padding_num)
                choice = np.hstack([shuffle_choice, padding_choice])
            else:
                choice = np.arange(stop=cur_num_points)
                last_num = padding_num
                while last_num > cur_num_points:
                    padding_choice = np.arange(stop=cur_num_points)
                    choice = np.hstack([choice, padding_choice])
                    last_num -= cur_num_points
                if last_num != 0:
                    padding_choice = np.arange(stop=last_num)
                    choice = np.hstack([choice, padding_choice])
            current_points = current_points[choice, :]
            current_points_labels = current_points_labels[choice]
            mask = torch.cat([torch.ones(cur_num_points), torch.zeros(padding_num)]).type(torch.int32)

        label = torch.from_numpy(dataset_dict[kfg.LABELS]).type(torch.int64)
        current_points_labels = torch.from_numpy(current_points_labels).type(torch.int64)
        if self.transforms is not None:
            current_points = self.transforms(current_points)
        features = get_part_seg_features(self.infeats_dim, current_points)

        ret = {
            kfg.POINTS: current_points,
            kfg.FEATS: features,
            kfg.POINTS_LABELS: current_points_labels,
            kfg.MASKS: mask,
            kfg.LABELS: label,
        }
        dict_as_tensor(ret)

        return ret