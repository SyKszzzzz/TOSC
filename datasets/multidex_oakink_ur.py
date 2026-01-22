from typing import Any, Tuple, Dict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../'))
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf

from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
from datasets.transforms import make_default_transform
from datasets.base import DATASET
import trimesh
import trimesh as tm
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
import json
import os
from collections import namedtuple

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from oikit.oi_shape import OakInkShape
from oikit.oi_shape.utils import ALL_INTENT
from yacs.config import CfgNode as CN

from datasets.grasp_data import GraspData
from datasets.grasp_query import Queries
import open3d as o3d

from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_farthest_points



OIShapeGrasp = namedtuple(
    "OIShapeGrasp",
    [
        "split",
        "sample_idx",
        "obj_id",
        "contact_region",
        "contactness",
        "obj_verts_obj_processed",
        "obj_rot",
        "obj_transl",
        "sbj_name",
        "hand_verts_obj",
        "joints_obj",
        "hand_pose_obj",
        "hand_shape",
    ],
)
OIShapeGrasp.__new__.__defaults__ = (None,) * len(OIShapeGrasp._fields)






x_upper = torch.tensor([ 2.8045e+00,  2.7018e+00,  3.0458e+00,  3.4708e-01,  5.0742e-01,
         1.6540e+00,  1.9258e-01,  3.0968e-01,  1.6662e+00,  4.1818e-01,
         2.6016e-01,  1.6399e+00,  1.4105e-01,  7.8212e-01,  1.7043e+00,
         1.3340e-01,  2.0309e-01,  1.6078e+00,  1.4439e-02,  2.0001e-01,
         1.7031e+00, -2.2862e-02,  8.9844e-01,  1.4268e+00,  7.4993e-02,
         2.0151e-01,  1.4444e+00, -9.0373e-05,  1.7854e-01,  1.4866e+00,
         1.0868e-01,  7.0282e-01,  1.6982e+00,  7.9677e-02,  3.2444e-01,
         1.6026e+00, -6.1725e-05,  2.0633e-01,  1.6037e+00,  1.4480e+00,
         4.9252e-01,  1.1986e+00,  8.1752e-01,  5.8521e-01,  1.0878e+00,
         6.4195e-01,  9.0342e-02,  6.0882e-01,  3.8521e-02,  1.3721e-02,
         2.2500e-02,  3.5441e-02,  2.0428e-02,  3.8304e-02,  1.9818e-02,
         1.3290e-02,  2.0268e-02,  1.8080e-02,  1.4512e-01,  1.2998e-01,
         1.7509e-01])


x_lower = torch.tensor([-2.8087e+00, -2.7107e+00, -3.0719e+00, -2.7644e-01, -8.1655e-01,
         2.4890e-04, -4.2917e-01, -2.1190e-01, -2.6136e-03, -6.4922e-02,
        -1.9199e-01,  1.1939e-04, -6.2365e-01, -7.9654e-01,  1.4647e-02,
        -4.1412e-01, -2.8861e-01,  1.8073e-04, -5.1573e-01, -1.8757e-01,
         1.4935e-04, -1.1803e+00, -6.7920e-01,  5.6621e-02, -1.0473e+00,
        -4.1329e-01,  7.6797e-05, -1.0065e+00, -4.8045e-01,  1.3998e-04,
        -6.3780e-01, -7.0535e-01,  7.0273e-03, -7.4499e-01, -4.0315e-01,
         1.4356e-04, -7.5175e-01, -2.4576e-01,  2.0090e-04, -5.6672e-01,
        -1.2392e+00, -2.6292e-01, -2.9719e-01, -7.6436e-01, -3.0848e-01,
        -8.5829e-03, -1.1928e+00, -1.1817e-01, -4.3088e-02, -3.4845e-02,
        -1.3116e-02, -2.0119e-02, -3.5874e-02, -2.6014e-02, -1.4266e-02,
        -8.3219e-03, -2.1315e-02, -2.7304e-02, -1.3843e-01, -1.1820e-01,
        -1.4180e-01])


@DATASET.register()
class OIShape(GraspData):

    def __init__(self, cfg, phase, slurm):
        self.transform = make_default_transform(cfg, phase) 
        super(OIShape, self).__init__(cfg, phase)

    def _preload(self):
        self.grasp_tuple = OIShapeGrasp
        os.environ["OAKINK_DIR"] = os.path.join(self.data_root, "OakInk")
        self.base_dataset = OakInkShape(
            category=self.cfg.OBJ_CATES,
            intent_mode=self.cfg.INTENT_MODE,
            data_split=self.phase,
            use_cache=self.use_cache,
            use_downsample_mesh=True,
            preload_obj=False,
        )
        self.action_id_to_intent = {v: k for k, v in ALL_INTENT.items()}

    def _init_grasp(self):
        self.grasp_list = self.base_dataset.grasp_list

    def _init_obj_warehouse(self):
        self.obj_warehouse = self.base_dataset.obj_warehouse  # 保存物体信息的

    def __len__(self):
        return len(self.base_dataset)

    def get_obj_id(self, idx):
        return self.grasp_list[idx]["obj_id"]

    def get_obj_verts(self, idx):
        return np.asarray(self.base_dataset.get_obj_mesh(idx).vertices, dtype=np.float32)

    def get_obj_faces(self, idx):
        return np.asarray(self.base_dataset.get_obj_mesh(idx).faces).astype(np.int32)
    
    def get_obj_pc(self, idx):
        return np.asarray(self.base_dataset.get_obj_pc(idx)).astype(np.int32)

    def get_obj_normals(self, idx):
        return np.asarray(self.base_dataset.get_obj_mesh(idx).vertex_normals, dtype=np.float32)

    def get_joints_obj(self, idx):
        return self.grasp_list[idx]["joints"]

    def get_hand_shape(self, idx):
        return self.grasp_list[idx]["hand_shape"]

    def get_hand_pose_obj(self, idx):
        return self.grasp_list[idx]["hand_pose"]

    def get_obj_rotmat(self, idx):
        return np.eye(3, dtype=np.float32)

    def get_intent(self, idx):
        act_id = self.grasp_list[idx]["action_id"]
        intent_name = self.action_id_to_intent[act_id]
        return int(act_id), intent_name

    def get_handover(self, idx):
        alt_j, alt_v, alt_pose, alt_shape, alt_tsl = self.base_dataset.get_hand_over(idx)
        return alt_j, alt_v, alt_pose, alt_shape, alt_tsl



    def read_color_info(self, color_info_path):
        color_to_semantic = {}
        with open(color_info_path, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    semantic_label = parts[0].strip()
                    rgb_values = parts[1].strip().split(',')
                    if len(rgb_values) == 3:
                        r, g, b = map(float, rgb_values)
                        # color = (r, g, b)  # Keep the RGB values in normalized form
                        color = (round(r, 3), round(g, 3), round(b, 3))
                        color_to_semantic[color] = semantic_label
        return color_to_semantic

    def read_pointcloud(self, ply_path):
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        colors = np.round(colors, 3)
        return points, colors
    

    


    def get_sample_identifier(self, idx):
        cate_id = self.grasp_list[idx]["cate_id"]
        obj_id = self.grasp_list[idx]["obj_id"]
        act_id = self.grasp_list[idx]["action_id"]
        intent_name = self.action_id_to_intent[act_id]
        subject_id = self.grasp_list[idx]["subject_id"]
        seq_ts = self.grasp_list[idx]["seq_ts"]
        return (f"{self.name}_{self.data_split}_CATE_{cate_id}"
                f"_OBJ({obj_id})_INT({intent_name})_SUB({subject_id})_TS({seq_ts})")
    # self.name = OIShape, 
    # eg:   OIShape_train_CATE_knife_OBJ(s20205)_INT(use)_SUB(0010)_TS(2021-10-04-15-44-21)

    def __getitem__(self, idx):
        if self.transform is not None:
            data = self.transform(super().__getitem__(idx), modeling_keys=None)
        return data
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)
    
    