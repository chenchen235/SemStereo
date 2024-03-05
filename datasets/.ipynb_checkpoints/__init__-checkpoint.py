from .kitti_dataset_1215 import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .us3d_dataset import Us3dDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "us3d": Us3dDataset
}
