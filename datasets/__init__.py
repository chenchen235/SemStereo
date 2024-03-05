# from .kitti_dataset_1215_aug import KITTIDataset
from .kitti_dataset_15 import KITTIDataset

# from .kitti_dataset_1215_augmentation import KITTIDataset

from .sceneflow_dataset_augmentation import SceneFlowDatset
# from .us3d_dataset import Us3dDataset
from .us3d_ import Us3dDataset
from .cityscapes_dataset_c import CityscapesDataset
# from .cityscapes_dataset import CityscapesDataset
from .whu_dataset import WhuDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "us3d": Us3dDataset,
    "cityscapes": CityscapesDataset,
    "WhuDataset": WhuDataset
}
