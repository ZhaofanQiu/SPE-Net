import torch
import numpy as np
import pytorchpoints.functional.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

def pc_normalize(pc):
    # Center and rescale point for 1m radius
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale
    return pc

def get_cls_features(input_features_dim, pc, normal=None):
    if input_features_dim == 3:
        features = pc
    elif input_features_dim == 4:
        features = torch.ones(size=(pc.shape[0], 1), dtype=torch.float32)
        features = torch.cat([features, pc], -1)
    elif input_features_dim == 6:
        features = torch.cat([pc, normal])
    elif input_features_dim == 7:
        features = torch.ones(size=(pc.shape[0], 1), dtype=torch.float32)
        features = torch.cat([features, pc, normal], -1)
    else:
        raise NotImplementedError("error")
    return features.transpose(0, 1).contiguous()

def get_part_seg_features(input_features_dim, pc):
    if input_features_dim == 3:
        features = pc
    elif input_features_dim == 4:
        features = torch.ones(size=(pc.shape[0], 1), dtype=torch.float32)
        features = torch.cat([features, pc], -1)
    else:
        raise NotImplementedError("error")
    return features.transpose(0, 1).contiguous()

def get_scene_seg_features(input_features_dim, pc, color, height):
    if input_features_dim == 1:
        features = height
    elif input_features_dim == 3:
        features = color
    elif input_features_dim == 4:
        features = torch.cat([color, height], -1)
    elif input_features_dim == 5:
        ones = torch.ones_like(height)
        features = torch.cat([ones, color, height], -1)
    elif input_features_dim == 6:
        features = torch.cat([color, pc], -1)
    elif input_features_dim == 7:
        features = torch.cat([color, height, pc], -1)
    else:
        raise NotImplementedError("error")
    return features.transpose(0, 1).contiguous()

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)