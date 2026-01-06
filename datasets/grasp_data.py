from abc import ABCMeta
from collections import namedtuple
from typing import Dict, List

import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.spatial.distance import cdist
from torch.utils.data._utils.collate import default_collate
from datasets.grasp_query import Queries, match_collate_queries

import logging
import os.path
import time
from typing import Optional
import json
from termcolor import colored

from typing import Optional
from tqdm import tqdm
from types import MethodType
import open3d as o3d
import torch
from torch import nn
import random
import torch.nn.functional as F
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_farthest_points
from torch.distributions.normal import Normal
MANOOutput = namedtuple(
    "MANOOutput",
    [
        "verts",
        "joints",
        "center_idx",
        "center_joint",
        "full_poses",
        "betas",
        "transforms_abs",
    ],
)
MANOOutput.__new__.__defaults__ = (None,) * len(MANOOutput._fields)




class PoseDisturber(nn.Module):

    def __init__(self, tsl_sigma=0.02, pose_sigma=0.2, root_rot_sigma=0.004, **kwargs):
        super().__init__()

        self.hand_transl_dist = Normal(torch.tensor(0.0), tsl_sigma)
        self.hand_pose_dist = Normal(torch.tensor(0.0), pose_sigma)
        self.hand_root_rot_dist = Normal(torch.tensor(0.0), root_rot_sigma)

    def forward(self, hand_pose, hand_transl):
        batch_size = hand_pose.shape[0]
        device = hand_pose.device

        hand_root_pose = hand_pose[:, :3]
        hand_rel_pose = hand_pose[:, 3:]

        hand_transl = hand_transl + self.hand_transl_dist.sample((batch_size, 3)).to(device)
        hand_root_pose = hand_root_pose + self.hand_root_rot_dist.sample((batch_size, 3)).to(device)
        hand_rel_pose = hand_rel_pose + self.hand_pose_dist.sample((batch_size, 15 * 3)).to(device)
        hand_pose = torch.cat([hand_root_pose, hand_rel_pose], dim=1)

        return hand_pose, hand_transl


def set_description(self, _: str):
    # if rank != 0, output nothing!
    pass


def etqdm(iterable, rank: Optional[int] = None, **kwargs):
    if rank:
        iterable.set_description = MethodType(set_description, iterable)
        return iterable
    else:
        return tqdm(iterable, bar_format="{l_bar}{bar:3}{r_bar}", colour="#ffa500", **kwargs)
    

class Formatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    time_str = "%(asctime)s"
    level_str = "[%(levelname)7s]"
    msg_str = "%(message)s"
    file_str = "(%(filename)s:%(lineno)d)"

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SteamFormatter(Formatter):

    FORMATS = {
        logging.DEBUG:
            colored(Formatter.msg_str, "cyan"),
        logging.INFO:
            colored(" ".join([Formatter.time_str, Formatter.level_str, ""]), "white", attrs=["dark"]) + \
            colored(Formatter.msg_str, "white"),
        logging.WARNING:
            colored(" ".join([Formatter.time_str, Formatter.level_str, ""]), "yellow", attrs=["dark"]) + \
            colored(Formatter.msg_str, "yellow"),
        logging.ERROR:
            colored(" ".join([Formatter.time_str, Formatter.level_str, ""]), "red", attrs=["dark"]) + \
            colored(Formatter.msg_str, "red") + colored(" " + Formatter.file_str, "red", attrs=["dark"]),
        logging.CRITICAL:
            colored(" ".join([Formatter.time_str, Formatter.level_str, ""]), "red", attrs=["dark", "bold"]) +\
            colored(Formatter.msg_str, "red", attrs=["bold"],) +\
            colored(" " + Formatter.file_str, "red", attrs=["dark", "bold"]),
    }


class FileFormatter(Formatter):

    FORMATS = {
        logging.INFO: " ".join([Formatter.time_str, Formatter.level_str, Formatter.msg_str]),
        logging.WARNING: " ".join([Formatter.time_str, Formatter.level_str, Formatter.msg_str]),
        logging.ERROR: " ".join([Formatter.time_str, Formatter.level_str, Formatter.msg_str, Formatter.file_str]),
        logging.CRITICAL: " ".join([Formatter.time_str, Formatter.level_str, Formatter.msg_str, Formatter.file_str]),
    }


class ExpLogger(logging.Logger):

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time.time()))
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        self.set_log_stream()
        self.filehandler = None


    def set_log_stream(self):
        self.stearmhandler = logging.StreamHandler()
        self.stearmhandler.setFormatter(SteamFormatter())
        self.stearmhandler.setLevel(logging.DEBUG)

        self.addHandler(self.stearmhandler)

    def remove_log_stream(self):
        self.removeHandler(self.stearmhandler)


    def set_log_file(self, path: str, name: Optional[str] = None):
        if self.filehandler is not None:
            self.warning("log file path can only be set once")
            return
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, f"{self.name}.log" if name is None else f"{name}.log")
        self.filehandler = logging.FileHandler(file_path)
        self.filehandler.setFormatter(FileFormatter())
        self.filehandler.setLevel(logging.INFO)
        self.addHandler(self.filehandler)


    def info(self, msg, **kwargs) -> None:
        return super().info(msg, **kwargs)


    def warning(self, msg, **kwargs) -> None:
        return super().warning(msg, **kwargs)


    def error(self, msg, **kwargs) -> None:
        return super().error(msg, **kwargs)

    def debug(self, msg, **kwargs) -> None:
        return super().debug(msg, **kwargs)

    def critical(self, msg, **kwargs) -> None:
        return super().critical(msg, **kwargs)


logger = ExpLogger()


def fps(data, number):
    data = data.squeeze(0)
    L0_point_cloud = Pointclouds(points=[torch.tensor(data, dtype=torch.float32)])

    sampled_points, sampled_indices = sample_farthest_points(L0_point_cloud.points_padded(), K=number)
    return sampled_points

def seprate_point_cloud(xyz, num_points, crop, fixed_points=None, padding_zeros=False):
    """
    seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    """
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        distance_matrix = torch.norm(
            center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1
        )

        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):

            
            INPUT.append(fps(input_data, 2048)[0])
            CROP.append(fps(crop_data, 2048)[0])
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)
    crop_data = torch.cat(CROP, dim=0)

    return input_data.contiguous(), crop_data.contiguous()

class GraspData(metaclass=ABCMeta):
    x_upper = torch.tensor([ 2.8045e+00,  2.7280e+00,  3.0458e+00,  3.1159e-01,  5.1721e-01,
         1.6539e+00,  2.2438e-01,  3.2031e-01,  1.6634e+00,  2.6552e-01,
         2.8611e-01,  1.6351e+00,  1.4105e-01,  7.8478e-01,  1.6985e+00,
         1.3340e-01,  2.0309e-01,  1.6071e+00,  1.2948e-02,  9.0582e-02,
         1.7031e+00, -8.1832e-03,  8.7749e-01,  1.4116e+00,  7.4993e-02,
         1.4705e-01,  1.4444e+00, -1.4037e-04,  1.7854e-01,  1.4866e+00,
         5.9179e-02,  7.0282e-01,  1.6982e+00,  7.9677e-02,  3.2444e-01,
         1.6026e+00,  2.7622e-02,  1.8142e-01,  1.5989e+00,  1.4053e+00,
         5.3830e-01,  1.1920e+00,  8.2044e-01,  5.8521e-01,  1.0868e+00,
         6.6499e-01,  1.6485e-01,  5.3404e-01,  8.6090e-02,  1.3721e-02,
        -5.7371e-02,  5.8875e-02,  2.0428e-02, -4.4386e-03,  1.5745e-02,
         1.0648e-01,  2.0268e-02,  1.8080e-02,  1.7642e-01,  1.4640e-01,
         6.5491e-02])
    x_lower = torch.tensor([-2.8087e+00, -2.6845e+00, -3.0719e+00, -3.1194e-01, -8.0673e-01,
         1.9884e-04, -3.9736e-01, -2.0127e-01, -5.4409e-03, -2.1758e-01,
        -1.6604e-01, -4.6595e-03, -6.2365e-01, -7.9386e-01,  8.7340e-03,
        -4.1412e-01, -2.8861e-01, -5.1892e-04, -5.1723e-01, -2.9699e-01,
         1.2898e-04, -1.1656e+00, -7.0016e-01,  4.1394e-02, -1.0473e+00,
        -4.6774e-01,  7.6771e-05, -1.0066e+00, -4.8045e-01,  1.3995e-04,
        -6.8730e-01, -7.0535e-01,  7.0274e-03, -7.4499e-01, -4.0315e-01,
         1.4353e-04, -7.2406e-01, -2.7067e-01, -4.5738e-03, -6.0943e-01,
        -1.1934e+00, -2.6951e-01, -2.9425e-01, -7.6436e-01, -3.0952e-01,
         1.4448e-02, -1.1183e+00, -1.9295e-01,  4.4823e-03, -3.4845e-02,
        -9.2987e-02,  3.3150e-03, -3.5874e-02, -6.8758e-02, -1.8340e-02,
         8.4872e-02, -2.1315e-02, -2.7304e-02, -1.0714e-01, -1.0177e-01,
        -2.5140e-01])
    
    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.
    def __init__(self, cfg, phase):
        super().__init__()
        self.name = self.__class__.__name__
        self.cfg = cfg
        self.phase = phase
        self.version = cfg.get("VERSION", "")
        self.data_root = cfg.DATA_ROOT
        self.data_split = phase
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.data_mode = cfg.get("DATA_MODE", "intent")
        self.pre_load_obj = cfg.get("PRE_LOAD_OBJ", False)
        self.use_cache = cfg.DATA_PRESET.get("USE_CACHE", False)
        self.filter_no_contact = cfg.DATA_PRESET.get("FILTER_NO_CONTACT", False)
        self.filter_no_contact_thresh = cfg.DATA_PRESET.get("FILTER_NO_CONTACT_THRESH", 5.0)
        self.n_points = cfg.DATA_PRESET.get("N_RESAMPLED_OBJ_POINTS", 2048)
        self.side = "right"
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side=self.side,
            center_idx=self.center_idx,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )

        self._preload()
        self._init_obj_warehouse()
        self._init_grasp()

    def _preload(self):
        self.grasp_tuple = namedtuple("MetaGrasp", None)

    def _init_grasp(self):
        self.grasp_list = list()

    def _init_obj_warehouse(self):
        self.obj_warehouse = dict()

    def _logging_info(self):
        logger.info(f"{self.name}-{self.data_split} has {len(self.grasp_list)} samples")

    def __len__(self):
        return len(self.grasp_list)

    def _filter_no_contact(self, grasp_list: List):
        nfiltered = 0

        filtered_grasp_list = []
        for _, g in enumerate(etqdm(grasp_list, desc=f"Filter {self.name} grasps")):
            grasp = self.grasp_tuple(**g)

            # filter no contact
            if self.filter_no_contact:
                # naive
                obj_verts_obj = self.obj_warehouse[grasp.obj_id].vertices  # (NVERTS, 3)
                joints_obj = grasp.joints_obj  # (21, 3)
                min_dist = np.min(cdist(obj_verts_obj, joints_obj) * 1000.0)
                if min_dist > self.filter_no_contact_thresh:
                    nfiltered += 1
                    continue

            filtered_grasp_list.append(grasp)
        return filtered_grasp_list

    def get_sample_identifier(self, idx):
        raise NotImplementedError

    def get_obj_id(self, idx):
        return self.grasp_list[idx].obj_id

    def get_joints_obj(self, idx):
        return self.grasp_list[idx].joints_obj

    def get_hand_shape(self, idx):
        return self.grasp_list[idx].hand_shape

    def get_hand_pose_obj(self, idx):
        return self.grasp_list[idx].hand_pose_obj

    def get_intent(self, idx):
        raise NotImplementedError(f"{self.name} not support intent")

    def get_handover(self, idx):
        raise NotImplementedError(f"{self.name} not support handover")

    def get_obj_rotmat(self, idx):
        return self.grasp_list[idx].obj_rot

    def get_obj_verts(self, idx):
        raise NotImplementedError()

    def get_obj_faces(self, idx):
        raise NotImplementedError()

    def get_obj_normals(self, idx):
        raise NotImplementedError()

    def process_obj_pack(self, obj_verts, obj_faces, n_sample_verts):
        mesh = Meshes(verts=torch.from_numpy(obj_verts).unsqueeze(0), faces=torch.from_numpy(obj_faces).unsqueeze(0))
        obj_verts_ds, obj_normals_ds = sample_points_from_meshes(mesh, n_sample_verts, return_normals=True)
        obj_pack = dict(
            obj_verts=obj_verts,
            obj_faces=obj_faces,
            obj_verts_ds=obj_verts_ds.squeeze(0).numpy(),
            obj_normals_ds=obj_normals_ds.squeeze(0).numpy()
            # ...
        )
        return obj_pack

    def process_hand_pack(self, pose, shape, joints_obj):
        joints_in_obj_sys = joints_obj
        mano_out = self.mano_layer(torch.from_numpy(pose[None, ...]), torch.from_numpy(shape[None, ...]))
        joints_in_mano_sys = mano_out.joints.squeeze(0).numpy()  # (21, 3)
        transl = np.mean(joints_in_obj_sys - joints_in_mano_sys, axis=0, keepdims=True)
        verts_in_obj_sys = mano_out.verts.squeeze(0) + transl  # (778, 3)

        hand_pack = dict(
            hand_joints=joints_in_obj_sys,
            hand_verts=verts_in_obj_sys,
            hand_transl=transl,
            hand_pose=pose,
            hand_shape=shape,
            # ...
        )
        return hand_pack

    def __getitem__(self, idx):
        if self.data_mode == "obj":
            return self.getitem_obj(idx)
        elif self.data_mode == "grasp":
            return self.getitem_grasp(idx)
        elif self.data_mode == "intent":
            return self.getitem_intent(idx)
        elif self.data_mode == "handover":
            return self.getitem_handover(idx)
        elif self.data_mode == "partial":
            return self.getitem_partial(idx)
            
        else:
            raise ValueError(f"Unknown data mode {self.data_mode}")

    def getitem_obj(self, idx):
        sample = {}

        sample[Queries.SAMPLE_IDENTIFIER] = self.get_sample_identifier(idx) 
        sample[Queries.OBJ_ID] = self.get_obj_id(idx) 
        obj_verts_obj = self.get_obj_verts(idx)
        obj_faces = self.get_obj_faces(idx)

        proc_obj = self.process_obj_pack(obj_verts_obj, obj_faces, self.n_points) 
        obj_verts_obj_ds = proc_obj["obj_verts_ds"]  
        obj_normals_obj_ds = proc_obj["obj_normals_ds"]  


        obj_rotmat = self.get_obj_rotmat(idx)

        point_cloud_center = obj_verts_obj_ds.mean(axis=0)
        centralized_point_cloud = obj_verts_obj_ds - point_cloud_center


        sample.update({
            Queries.OBJ_VERTS_OBJ: obj_verts_obj, 
            Queries.OBJ_VERTS_OBJ_DS_ORI: centralized_point_cloud, 
            Queries.OBJ_NORMALS_OBJ_DS: obj_normals_obj_ds,
            Queries.OBJ_FACES: obj_faces, 
            Queries.OBJ_ROTMAT: obj_rotmat, 
        })
        return sample
    
    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self.x_lower), (self.x_upper - self.x_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self.x_upper - self.x_lower) + self.x_lower
        return global_trans_denorm
    def getitem_grasp(self, idx):
        sample = self.getitem_obj(idx)
        joints_obj = self.get_joints_obj(idx) 
        hand_objcet_trans = joints_obj[9,:]
        hand_pose_obj = self.get_hand_pose_obj(idx)
        hand_shape = self.get_hand_shape(idx)

        hand_para = np.concatenate((hand_pose_obj, hand_shape, hand_objcet_trans))
        hand_para_tensor = torch.tensor(hand_para, dtype=torch.float32).cpu()
        hand_para_tensor = self.trans_normalize(hand_para_tensor)
        sample.update({
            Queries.JOINTS_OBJ: joints_obj, 
            Queries.HAND_POSE_OBJ: hand_pose_obj,
            Queries.HAND_SHAPE: hand_shape,  
            Queries.HAND_PARA: hand_para_tensor
        })
        return sample

    def getitem_intent(self, idx):
        sample = self.getitem_grasp(idx)
        intent_id, intent_name = self.get_intent(idx)
        sample.update({
            Queries.INTENT_ID: intent_id,  
            Queries.INTENT_NAME: intent_name,  
        })
        return sample
    
    def getitem_guidence(self, idx):
        sample = self.getitem_grasp(idx)
        file_path = self.grasp_list[idx]["file_path"]
        ins_path = file_path.replace('shape/oakink_shape_v2', 'CapGrasp/captions/v0_1')
        ins_path = os.path.dirname(ins_path)
        all_files = os.listdir(ins_path)
        simple_caption_files = [f for f in all_files if f.endswith('_simple_caption.json')]
        num_files = len(simple_caption_files)
        ins_id = np.random.randint(0, num_files)
        ins_json = simple_caption_files[ins_id]
        with open(os.path.join(ins_path, ins_json), 'r') as f:
            data = json.load(f)
        for entry in data:
            if entry.get("from") == "human":
                ins = entry.get("value")
        sample.update({
            Queries.GUIDENCE: ins,  
            Queries.INS_PATH: ins_path,
            
        })
        return sample

    
    def getitem_partial(self, idx):

        sample_identifier = self.get_sample_identifier(idx)
        sample = self.getitem_grasp(idx)        
        intent_id, intent_name = self.get_intent(idx)        
        complete_path = "/home/syks/Scene-Diffuser-obj/data/OakInk/complete"
        file_path = os.path.join(complete_path,f"['{sample_identifier}']")
        partial_pcd_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/complete/['{sample_identifier}']/complete.npy"
        sampled_points = np.load(partial_pcd_path)
        partial_pcd_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/complete/['{sample_identifier}']/partial_point_cloud.ply"
        partial_pcd = o3d.io.read_point_cloud(partial_pcd_path)
        partial_points = np.asarray(partial_pcd.points)
        json_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/complete/['{sample_identifier}']/viewpoint.json"
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        viewpoint = json_file[-1]['final_viewpoint'][0]
        guidence = json_file[-1]['guidence']
        
        sample.update({
            Queries.OBJ_VERTS_OBJ_DS: sampled_points,  
            Queries.OBJ_VERTS_OBJ_DS_ORI:partial_points, 
            Queries.GUIDENCE: guidence,
            Queries.INTENT_ID: intent_id,  
            Queries.INTENT_NAME: intent_name,  
            Queries.VIEWPOINT: viewpoint,  
        })
        return sample
    
    def getitem_handover(self, idx):
        sample = self.getitem_grasp(idx)
        alt_j, alt_v, alt_pose, alt_shape, _ = self.get_handover(idx)
        sample.update({
            Queries.ALT_JOINTS_OBJ: alt_j,
            Queries.ALT_HAND_POSE_OBJ: alt_pose,
            Queries.ALT_HAND_SHAPE: alt_shape,
            Queries.ALT_HAND_VERTS_OBJ: alt_v,
        })
        return sample


def grasp_data_collate(batch: List[Dict]):
    """
    Collate function, duplicating the items in extend_queries along the
    first dimension so that they all have the same length.
    Typically applies to faces and vertices, which have different sizes
    depending on the object.
    """
    # *  NEW QUERY: CollateQueries.PADDING_MASK

    extend_queries = {Queries.OBJ_VERTS_OBJ, Queries.OBJ_NORMALS_OBJ, Queries.OBJ_FACES}
    pop_queries = []
    for poppable_query in extend_queries:
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    # Remove fields that don't have matching sizes
    for pop_query in pop_queries:
        padding_query_field = match_collate_queries(pop_query)
        max_size = max([sample[pop_query].shape[0] for sample in batch])
        for sample in batch:
            pop_value = sample[pop_query]
            orig_len = pop_value.shape[0]
            # Repeat vertices so all have the same number
            pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
            sample[pop_query] = pop_value
            if padding_query_field not in sample:
                # generate a new field, contains padding mask
                # note that only the beginning pop_value.shape[0] points are in effect
                # so the mask will be a vector of length max_size, with origin_len ones in the beginning
                padding_mask = np.zeros(max_size, dtype=np.int32)
                padding_mask[:orig_len] = 1
                sample[padding_query_field] = padding_mask

    # store the mask filtering the points
    batch = default_collate(batch)  # this function np -> torch
    return batch
