from .build import (
    build_points_train_loader,
    build_points_test_loader,
)

from .common import DatasetFromList, MapDataset
from .ModelNet40 import ModelNet40Cls
from .PartNet import PartNetSeg
from .S3DIS import S3DISSeg
from .ShapeNetPart import ShapeNetPartSeg