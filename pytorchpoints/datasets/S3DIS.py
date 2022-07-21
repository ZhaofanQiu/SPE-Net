import os
import copy
import numpy as np
import pickle
import h5py
import torch
import torchvision
from pytorchpoints.functional import pc_normalize, get_scene_seg_features, dict_as_tensor, grid_subsampling
from pytorchpoints.config import configurable
from pytorchpoints.config import kfg
from .build import DATASETS_REGISTRY
from .transforms import PointcloudScaleAndJitter, PointcloudToTensor, PointcloudRandomRotate
from sklearn.neighbors import KDTree

__all__ = ["S3DISSeg"]

def get_label_to_names():
    label_to_names = {  0: 'ceiling',   1: 'floor',        2: 'wall',      3: 'beam',
                        4: 'column',    5: 'window',       6: 'door',      7: 'chair',
                        8: 'table',     9: 'bookcase',    10: 'sofa',     11: 'board',
                       12: 'clutter' }
    return label_to_names


@DATASETS_REGISTRY.register()
class S3DISSeg:
    @configurable
    def __init__(
        self,
        stage: str,
        data_root: str,
        infeats_dim: int,
        subsampling: float,
        color_drop: float,
        in_radius: float,
        num_points: int,
        num_steps: int,
        num_epochs: int,
        transforms,
    ):
        self.stage = stage
        self.data_root = data_root
        self.epoch = 0
        self.infeats_dim = infeats_dim
        self.transforms = transforms
        self.subsampling = subsampling
        self.color_drop = color_drop
        self.in_radius = in_radius
        self.num_points = num_points
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.label_to_names = get_label_to_names()
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.train_clouds = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
        #self.train_clouds = ['Area_1']
        self.val_clouds = ['Area_5']
        if self.stage == 'train':
            self.cloud_names = self.train_clouds
        else:
            self.cloud_names = self.val_clouds
      
        self.color_mean = np.array([0.5136457, 0.49523646, 0.44921124])
        self.color_std = np.array([0.18308958, 0.18415008, 0.19252081])

        self.folder = 'S3DIS'
        self.data_dir = os.path.join(self.data_root, self.folder, 'Stanford3dDataset_v1.2', 'processed')
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = {
            "stage": stage,
            "data_root": cfg.DATALOADER.DATA_ROOT,
            "infeats_dim": cfg.MODEL.INFEATS_DIM,
            "subsampling": cfg.DATALOADER.SUBSAMPLING,
            "color_drop": cfg.DATALOADER.COLOR_DROP,
            "in_radius": cfg.DATALOADER.IN_RADIUS,
            "num_points": cfg.DATALOADER.NUM_POINTS,
            "num_steps": cfg.DATALOADER.NUM_STEPS,
            "num_epochs": cfg.SOLVER.EPOCH,
        }

        if stage == "train":
            transforms = torchvision.transforms.Compose([
                PointcloudToTensor(),
                PointcloudRandomRotate(cfg.AUG.X_ANGLE_RANGE, cfg.AUG.Y_ANGLE_RANGE, cfg.AUG.Z_ANGLE_RANGE),
                PointcloudScaleAndJitter(
                    scale_low=cfg.AUG.SCALE_LOW, scale_high=cfg.AUG.SCALE_HIGH,
                    std=cfg.AUG.NOISE_STD, clip=cfg.AUG.NOISE_CLIP,
                    augment_symmetries=cfg.AUG.AUGMENT_SYMMETRIES)
            ])
        else:
            transforms = torchvision.transforms.Compose([PointcloudToTensor()])
        ret.update({ "transforms": transforms })
        return ret

    def load_data(self, cfg):
        datalist = list(range(cfg.DATALOADER.NUM_STEPS))

        # prepare data
        filename = os.path.join(self.data_dir, f'{self.stage}_{self.subsampling:.3f}_data.pkl')
        if not os.path.exists(filename):
            cloud_points_list = []
            cloud_points_color_list = []
            cloud_points_label_list = []

            sub_cloud_points_list = []
            sub_cloud_points_color_list = []
            sub_cloud_points_label_list = []
            sub_cloud_tree_list = []
            for cloud_idx, cloud_name in enumerate(self.cloud_names):
                cloud_file = os.path.join(self.data_dir, cloud_name + '.pkl')
                if os.path.exists(cloud_file):
                    with open(cloud_file, 'rb') as f:
                        cloud_points, cloud_colors, cloud_classes = pickle.load(f)
                else:
                    cloud_folder = os.path.join(self.data_root, self.folder, 'Stanford3dDataset_v1.2', cloud_name)
                    room_folders = [os.path.join(cloud_folder, room) for room in os.listdir(cloud_folder) if
                                    os.path.isdir(os.path.join(cloud_folder, room))]

                    cloud_points = np.empty((0, 3), dtype=np.float32)
                    cloud_colors = np.empty((0, 3), dtype=np.float32)
                    cloud_classes = np.empty((0, 1), dtype=np.int32)
                    for i, room_folder in enumerate(room_folders):
                        print('Cloud %s - Room %d/%d : %s' % (
                            cloud_name, i + 1, len(room_folders), room_folder.split('\\')[-1]))
                        
                        for object_name in os.listdir(os.path.join(room_folder, 'Annotations')):
                            if object_name[-4:] == '.txt':
                                object_file = os.path.join(room_folder, 'Annotations', object_name)
                                tmp = object_name[:-4].split('_')[0]
                                if tmp in self.name_to_label:
                                    object_class = self.name_to_label[tmp]
                                elif tmp in ['stairs']:
                                    object_class = self.name_to_label['clutter']
                                else:
                                    raise ValueError('Unknown object name: ' + str(tmp))

                                with open(object_file, 'r') as f:
                                    object_data = np.array([[float(x) for x in line.split()] for line in f])
                                cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
                                cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                                object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                                cloud_classes = np.vstack((cloud_classes, object_classes))
                    with open(cloud_file, 'wb') as f:
                        pickle.dump((cloud_points, cloud_colors, cloud_classes), f)
                
                cloud_points_list.append(cloud_points)
                cloud_points_color_list.append(cloud_colors)
                cloud_points_label_list.append(cloud_classes)

                sub_cloud_file = os.path.join(self.data_dir, cloud_name + f'_{self.subsampling:.3f}_sub.pkl')
                if os.path.exists(sub_cloud_file):
                    with open(sub_cloud_file, 'rb') as f:
                        sub_points, sub_colors, sub_labels, search_tree = pickle.load(f)
                else:
                    if self.subsampling > 0:
                        sub_points, sub_colors, sub_labels = grid_subsampling(cloud_points,
                                                                              features=cloud_colors,
                                                                              labels=cloud_classes,
                                                                              sampleDl=self.subsampling)
                        sub_colors /= 255.0
                        sub_labels = np.squeeze(sub_labels)
                    else:
                        sub_points = cloud_points
                        sub_colors = cloud_colors / 255.0
                        sub_labels = cloud_classes

                    # Get chosen neighborhoods
                    search_tree = KDTree(sub_points, leaf_size=50)

                    with open(sub_cloud_file, 'wb') as f:
                        pickle.dump((sub_points, sub_colors, sub_labels, search_tree), f)

                sub_cloud_points_list.append(sub_points)
                sub_cloud_points_color_list.append(sub_colors)
                sub_cloud_points_label_list.append(sub_labels)
                sub_cloud_tree_list.append(search_tree)

            self.clouds_points = cloud_points_list
            if self.stage != 'train':
                self.clouds_points_labels = cloud_points_label_list
            else:
                self.clouds_points_labels = 0
            self.sub_clouds_points = sub_cloud_points_list
            self.sub_clouds_points_colors = sub_cloud_points_color_list
            self.sub_clouds_points_labels = sub_cloud_points_label_list
            self.sub_cloud_trees = sub_cloud_tree_list
            with open(filename, 'wb') as f:
                pickle.dump((self.clouds_points, self.clouds_points_labels, self.sub_clouds_points, self.sub_clouds_points_colors, self.sub_clouds_points_labels,
                             self.sub_cloud_trees), f)
                print(f"{filename} saved successfully")

        else:
            with open(filename, 'rb') as f:
                (self.clouds_points, self.clouds_points_labels, self.sub_clouds_points, self.sub_clouds_points_colors, self.sub_clouds_points_labels,
                 self.sub_cloud_trees) = pickle.load(f)
                print(f"{filename} loaded successfully")

        # prepare iteration indices
        filename = os.path.join(self.data_dir, f'{self.stage}_{self.subsampling:.3f}_{self.num_epochs}_{self.num_steps}_iterinds.pkl')
        if not os.path.exists(filename):
            potentials = []
            min_potentials = []
            for cloud_i, tree in enumerate(self.sub_cloud_trees):
                print(f"{self.stage}/{cloud_i} has {tree.data.shape[0]} points")
                cur_potential = np.random.rand(tree.data.shape[0]) * 1e-3
                potentials.append(cur_potential)
                min_potentials.append(float(np.min(cur_potential)))
            self.cloud_inds = []
            self.point_inds = []
            self.noise = []
            for ep in range(self.num_epochs):
                print(ep)
                for st in range(self.num_steps):
                    cloud_ind = int(np.argmin(min_potentials))
                    point_ind = np.argmin(potentials[cloud_ind])
                    #print(f"[{ep}/{st}]: {cloud_ind}/{point_ind}")
                    self.cloud_inds.append(cloud_ind)
                    self.point_inds.append(point_ind)
                    points = np.array(self.sub_cloud_trees[cloud_ind].data, copy=False)
                    center_point = points[point_ind, :].reshape(1, -1)
                    noise = np.random.normal(scale=self.in_radius / 10, size=center_point.shape)
                    self.noise.append(noise)
                    pick_point = center_point + noise.astype(center_point.dtype)
                    # Indices of points in input region
                    query_inds = self.sub_cloud_trees[cloud_ind].query_radius(pick_point,
                                                                              r=self.in_radius,
                                                                              return_distance=True,
                                                                              sort_results=True)[0][0]
                    cur_num_points = query_inds.shape[0]
                    if self.num_points < cur_num_points:
                        query_inds = query_inds[:self.num_points]
                    # Update potentials (Tuckey weights)
                    dists = np.sum(np.square((points[query_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(self.in_radius))
                    tukeys[dists > np.square(self.in_radius)] = 0
                    potentials[cloud_ind][query_inds] += tukeys
                    min_potentials[cloud_ind] = float(np.min(potentials[cloud_ind]))
                    # print(f"====>potentials: {potentials}")
                    #print(f"====>min_potentials: {min_potentials}")
            with open(filename, 'wb') as f:
                pickle.dump((self.cloud_inds, self.point_inds, self.noise), f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.cloud_inds, self.point_inds, self.noise = pickle.load(f)
                print(f"{filename} loaded successfully")

        # prepare validation projection inds
        filename = os.path.join(self.data_dir, f'{self.stage}_{self.subsampling:.3f}_proj.pkl')
        if not os.path.exists(filename):
            proj_ind_list = []
            for points, search_tree in zip(self.clouds_points, self.sub_cloud_trees):
                proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                proj_inds = proj_inds.astype(np.int32)
                proj_ind_list.append(proj_inds)
            self.projections = proj_ind_list
            with open(filename, 'wb') as f:
                pickle.dump(self.projections, f)
                print(f"{filename} saved successfully")
        else:
            with open(filename, 'rb') as f:
                self.projections = pickle.load(f)
                print(f"{filename} loaded successfully")

        print('finish S3D')
        return datalist

    def __call__(self, dataset_dict):
        idx = dataset_dict
        cloud_ind = self.cloud_inds[idx + self.epoch * self.num_steps]
        point_ind = self.point_inds[idx + self.epoch * self.num_steps]
        noise = self.noise[idx + self.epoch * self.num_steps]
        points = np.array(self.sub_cloud_trees[cloud_ind].data, copy=False)
        center_point = points[point_ind, :].reshape(1, -1)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Indices of points in input region
        query_inds = self.sub_cloud_trees[cloud_ind].query_radius(pick_point,
                                                                  r=self.in_radius,
                                                                  return_distance=True,
                                                                  sort_results=True)[0][0]
        # Number collected
        cur_num_points = query_inds.shape[0]
        if self.num_points < cur_num_points:
            # choice = np.random.choice(cur_num_points, self.num_points)
            # input_inds = query_inds[choice]
            shuffle_choice = np.random.permutation(np.arange(self.num_points))
            input_inds = query_inds[:self.num_points][shuffle_choice]
            mask = torch.ones(self.num_points).type(torch.int32)
        else:
            shuffle_choice = np.random.permutation(np.arange(cur_num_points))
            query_inds = query_inds[shuffle_choice]
            padding_choice = np.random.choice(cur_num_points, self.num_points - cur_num_points)
            input_inds = np.hstack([query_inds, query_inds[padding_choice]])
            mask = torch.zeros(self.num_points).type(torch.int32)
            mask[:cur_num_points] = 1

        original_points = points[input_inds]
        current_points = (original_points - pick_point).astype(np.float32)
        current_points_height = original_points[:, 2:]
        current_points_height = torch.from_numpy(current_points_height).type(torch.float32)

        current_colors = self.sub_clouds_points_colors[cloud_ind][input_inds]
        current_colors = (current_colors - self.color_mean) / self.color_std
        current_colors = torch.from_numpy(current_colors).type(torch.float32)

        current_colors_drop = (torch.rand(1) > self.color_drop).type(torch.float32)
        current_colors = (current_colors * current_colors_drop).type(torch.float32)
        current_points_labels = torch.from_numpy(self.sub_clouds_points_labels[cloud_ind][input_inds]).type(torch.int64)
        current_cloud_index = torch.from_numpy(np.array(cloud_ind)).type(torch.int64)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        features = get_scene_seg_features(self.infeats_dim, current_points, current_colors,
                                          current_points_height)
        ret = {
            kfg.POINTS: current_points,
            kfg.MASKS: mask,
            kfg.FEATS: features,
            kfg.POINTS_LABELS: current_points_labels,
            kfg.CLOUD_INDEX: current_cloud_index,
            kfg.INPUT_INDEX: input_inds
        }
        dict_as_tensor(ret)
        return ret