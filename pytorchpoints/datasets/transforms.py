import os
import sys
import torch
import numpy as np


class PointcloudScaleAndJitter(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., std=0.01, clip=0.05, augment_symmetries=[0, 0, 0]):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.std = std
        self.clip = clip
        self.augment_symmetries = augment_symmetries

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        symmetries = np.round(np.random.uniform(low=0, high=1, size=[3])) * 2 - 1
        symmetries = symmetries * np.array(self.augment_symmetries) + (1 - np.array(self.augment_symmetries))
        xyz1 *= symmetries
        xyz2 = np.clip(np.random.normal(scale=self.std, size=[pc.shape[0], 3]), a_min=-self.clip, a_max=self.clip)
        pc[:, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(
            xyz2).float()

        return pc


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudRandomRotate(object):
    def __init__(self, x_range=np.pi, y_range=np.pi, z_range=np.pi):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def _get_angles(self):
        x_angle = np.random.uniform(-self.x_range, self.x_range)
        y_angle = np.random.uniform(-self.y_range, self.y_range)
        z_angle = np.random.uniform(-self.z_range, self.z_range)

        return np.array([x_angle, y_angle, z_angle])

    def __call__(self, points):
        if self.x_range == 0.0 and self.y_range == 0.0 and self.z_range == 0.0:
            return points

        angles_ = self._get_angles()
        Rx = angle_axis(angles_[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles_[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles_[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudTestRotate(object):
    def __init__(self, r_type='None'):
        self.r_type = r_type
        if r_type == 'Z':
            self.rotate = PointcloudRandomRotate(x_range=0, y_range=0, z_range=np.pi)
        elif r_type == 'SO3':
            self.rotate = PointcloudRandomRotate(x_range=np.pi, y_range=np.pi, z_range=np.pi)

    def __call__(self, points):
        if self.r_type == 'None':
            return points
        elif self.r_type in ('Z', 'SO3'):
            return self.rotate(points)
        else:
            raise NotImplementedError
