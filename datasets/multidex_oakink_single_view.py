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

# from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
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


OBJ_CLASS = [
    "bottle",
    "bowl",
    "cameras",
    "cup",
    "cylinder_bottle",
    "donut",
    "eyeglasses",
    "headphones",
    "knife",
    "lotion_pump",
    "mug",
    "pen",
    "pincer",
    "power_drill",
    "scissors",
    "squeezable",
    "trigger_sprayer",
    "wrench",
]


x_lower = torch.tensor([-2.8087e+00, -2.7107e+00, -3.0719e+00, -2.7644e-01, -8.1655e-01,
         2.4890e-04, -4.2917e-01, -2.2007e-01, -2.6136e-03, -6.7771e-02,
        -1.9199e-01,  1.1939e-04, -6.2365e-01, -7.9654e-01,  1.4647e-02,
        -4.1412e-01, -5.8028e-01,  1.8073e-04, -5.1573e-01, -1.9574e-01,
         1.2893e-04, -1.1803e+00, -6.7920e-01,  5.8778e-03, -1.0473e+00,
        -4.1329e-01,  7.6797e-05, -1.0065e+00, -4.8045e-01,  1.3998e-04,
        -6.3780e-01, -7.0535e-01,  7.0273e-03, -7.4499e-01, -4.0315e-01,
         1.4356e-04, -7.7549e-01, -2.4576e-01,  1.0434e-04, -5.6672e-01,
        -1.2392e+00, -2.6292e-01, -2.9719e-01, -7.6436e-01, -3.0848e-01,
        -5.3309e-02, -1.1928e+00, -1.1817e-01, -4.7334e-02, -3.4845e-02,
        -1.3147e-02, -2.1503e-02, -3.5874e-02, -2.6014e-02, -1.4266e-02,
        -8.3219e-03, -2.1315e-02, -2.7304e-02, -1.4562e-01, -1.1820e-01,
        -1.4180e-01])
x_upper = torch.tensor([ 2.8045e+00,  2.8593e+00,  3.0458e+00,  3.7927e-01,  5.2747e-01,
         1.6540e+00,  1.9258e-01,  3.0968e-01,  1.6662e+00,  5.0568e-01,
         2.6016e-01,  1.6478e+00,  1.4105e-01,  7.8212e-01,  1.7043e+00,
         1.3340e-01,  2.0309e-01,  1.6089e+00,  1.6088e-02,  2.0297e-01,
         1.7031e+00, -2.8203e-03,  8.9844e-01,  1.4268e+00,  7.4993e-02,
         2.0151e-01,  1.4444e+00, -9.0373e-05,  1.7854e-01,  1.4866e+00, 
         1.5842e-01,  7.0282e-01,  1.6982e+00,  7.9677e-02,  3.2444e-01,
         1.6026e+00, -3.1889e-05,  2.2176e-01,  1.6037e+00,  1.4480e+00,
         4.9252e-01,  1.1986e+00,  8.1752e-01,  5.8521e-01,  1.0878e+00,
         6.4195e-01,  1.0558e-01,  6.0882e-01,  3.8521e-02,  1.3721e-02,
         2.5672e-02,  3.5441e-02,  2.0428e-02,  4.1258e-02,  1.9818e-02,
         1.3290e-02,  2.0268e-02,  1.8080e-02,  1.4512e-01,  1.5682e-01,
         1.7509e-01])

@DATASET.register() 
class OIShape_SV(GraspData):

    def __init__(self, cfg, phase, slurm):
        self.transform = make_default_transform(cfg, phase) 
        super(OIShape_SV, self).__init__(cfg, phase)

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
    
    def get_llm_SV(self, idx):
        cate_id = self.grasp_list[idx]["cate_id"]      # 物体类型
        # act_id = self.grasp_list[idx]["action_id"] 
        # intent_name = self.action_id_to_intent[act_id] # 意图
        # 'file_path': '/home/syks/Scene-Diffuser-obj/data/OakInk/shape/oakink_shape_v2/trigger_sprayer/O01000/f655758323/s01101/hand_param.pkl'
        # ins_path : ''/home/syks/Scene-Diffuser-obj/data/OakInk/CapGrasp/captions/v0_1/trigger_sprayer/O01000/f655758323/s01101/'
        file_path = self.grasp_list[idx]["file_path"]
        # 加载ins
        # ----------------------------------------------------------------------------------------------------------------
        ins_path = file_path.replace('shape/oakink_shape_v2', 'CapGrasp/captions/v0_1')
        # 统计simple_caption数量
        ins_path = os.path.dirname(ins_path)
        all_files = os.listdir(ins_path)
        simple_caption_files = [f for f in all_files if f.endswith('_word.npy')]
        num_files = len(simple_caption_files)
        # print("num_files  is ", num_files)
        ins_id = np.random.randint(0, num_files)

        ins_disc = np.load(os.path.join(ins_path, str(ins_id)+"_word.npy"))[0]
        ins_mask_disc = np.load(os.path.join(ins_path, str(ins_id)+"_mask.npy"))[0]

        # ins_disc = np.ones(3)
        # ins_mask_disc = np.ones(3)
        # ----------------------------------------------------------------------------------------------------------------

       

        # 下面搞的是加载原始的ins，bert处理后的应该是找到
        # ----------------------------------------------------------------------------------------------------------------
        # file_path = self.grasp_list[idx]["file_path"]
        # ins_path = file_path.replace('shape/oakink_shape_v2', 'CapGrasp/captions/v0_1')
        # # 统计simple_caption数量
        # ins_path = os.path.dirname(ins_path)
        # all_files = os.listdir(ins_path)
        # simple_caption_files = [f for f in all_files if f.endswith('_simple_caption.json')]
        # num_files = len(simple_caption_files)

        # ins_id = np.random.randint(0, num_files)

        # ins_json = simple_caption_files[ins_id]

        # with open(os.path.join(ins_path, ins_json), 'r') as f:
        #     data = json.load(f)
        # for entry in data:
        #     if entry.get("from") == "human":
        #         ins = entry.get("value")
        # ----------------------------------------------------------------------------------------------------------------

        # 加载物体理解
        # ----------------------------------------------------------------------------------------------------------------
        obj_base = "/home/syks/Scene-Diffuser-obj/data/OakInk/SV_llm/obj/"
        obj_name, _ = self.base_dataset.get_obj_name(idx)

        obj_path = os.path.join(obj_base, obj_name)
        sv_id = np.random.randint(0, 10)

        obj_id = np.random.randint(0, 10)
        obj_disc = np.load(os.path.join(obj_path, str(sv_id),  str(obj_id)+"_word.npy"))[0]
        
        obj_mask_disc = np.load(os.path.join(obj_path, str(sv_id),  str(obj_id)+"_mask.npy"))[0]
        # ----------------------------------------------------------------------------------------------------------------
        return ins_disc, ins_mask_disc, sv_id, obj_disc, obj_mask_disc, ins_path
        
        # llm_path = os.path.join(self.data_root, "OakInk","llm")
        # # ins
        # # 这里的ins并不固定为52了，因为不同的不一样
        # task_ins_id = np.random.randint(0, 52)

        # # 指令
        # ins_disc = np.load(os.path.join(llm_path, "ins_disc", cate_id,  , str(task_ins_id)+"_word.npy"))[0]
        # ins_mask_disc = np.load(os.path.join(llm_path, "ins_disc", cate_id, intent_name, str(task_ins_id)+"_mask.npy"))[0]


        # # task
        # task_id = np.random.randint(0, 10)
        # task_disc = np.load(os.path.join(llm_path, "task_part", cate_id, intent_name, str(task_id)+"_word.npy"))[0]
        # task_mask_disc = np.load(os.path.join(llm_path, "task_part", cate_id, intent_name, str(task_id)+"_mask.npy"))[0]

        
        # obj  最多10个part，那就pad到10，然后加mask
        # 对于物体的描述
        # 对于残缺点云id就是sv_id

        


        

        # return ins_disc, ins_mask_disc, task_disc, task_mask_disc, sv_id, obj_disc, obj_mask_disc
        # task_mask_disc = os.paht.join(llm_path, "ins_disc", cate_id, intent_name, str(task_ins_id)+"_mask.npy")


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
    
    def get_part_indices(self, colors, color_to_semantic):
        part_indices = {label: [] for label in color_to_semantic.values()}

        for i, color in enumerate(colors):
            color_tuple = tuple(color)
            if color_tuple in color_to_semantic:
                semantic_label = color_to_semantic[color_tuple]
                part_indices[semantic_label].append(i)

        return part_indices
    
    # def get_sampled_part_indices(self, sampled_indices, part_indices):
    #     sampled_part_indices = {label: [] for label in part_indices.keys()}

    #     for sampled_idx in sampled_indices:
    #         for part, indices in part_indices.items():
    #             if sampled_idx in indices:
    #                 sampled_part_indices[part].append(sampled_idx)
    #                 break

    #     return sampled_part_indices
    def get_sampled_part_indices(self, sampled_indices, part_indices):
        sampled_part_indices = {label: [] for label in part_indices.keys()}

        # Create a reverse mapping from original indices to sampled indices
        original_to_sampled = {original_idx: sampled_idx for sampled_idx, original_idx in enumerate(sampled_indices)}

        for part, indices in part_indices.items():
            for original_idx in indices:
                if original_idx in original_to_sampled:
                    sampled_part_indices[part].append(original_to_sampled[original_idx])

        return sampled_part_indices
    
    def get_pointcloud_w_part(self, idx):
        cate_id = self.grasp_list[idx]["cate_id"]      # 物体类型
        obj_id = self.grasp_list[idx]["obj_id"]        # 物体id
        act_id = self.grasp_list[idx]["action_id"] 
        intent_name = self.action_id_to_intent[act_id] # 意图
        # pc = np.asarray(self.base_dataset.get_obj_pc_path(idx)).astype(np.int32)
        raw_pc_path, obj_name = self.base_dataset.get_obj_pc_path(idx)  # 带RGB的点云

        print("obj_name is ", obj_name)
        part_path = os.path.join(self.data_root, "OakInk","shape", "part")
        obj_part_path = os.path.join(part_path, obj_name)

        L0_path = os.path.join(obj_part_path, "L0")
        L1_path = os.path.join(obj_part_path, "L1")

        L0_color_info_path = os.path.join(L0_path, "color_info.txt")
        L0_ply_path = os.path.join(L0_path, "semantic_seg_all.ply")
        
        L1_color_info_path = os.path.join(L1_path, "color_info.txt")
        L1_ply_path = os.path.join(L1_path, "semantic_seg_all.ply")

        # Read point clouds for L0 and L1
        L0_color_to_semantic = self.read_color_info(L0_color_info_path)
        L1_color_to_semantic = self.read_color_info(L1_color_info_path)
        # print("L0_color_to_semantic is ", L0_color_to_semantic)
        # print("L1_color_to_semantic is ", L1_color_to_semantic)


        points, _ = self.read_pointcloud(raw_pc_path)
        _, L0_colors = self.read_pointcloud(L0_ply_path)
        # print("points shape is ", points.shape)
        # print("L0_colors shape is ", L0_colors.shape)
        # print("L0_colors is ", L0_colors)

        _, L1_colors = self.read_pointcloud(L1_ply_path)


        L0_point_indices = self.get_part_indices(L0_colors, L0_color_to_semantic)
        L1_point_indices = self.get_part_indices(L1_colors, L1_color_to_semantic)
        # print("L0_point_indices is ", L0_point_indices)
        # print("L1_point_indices is ", L1_point_indices)

        L0_point_cloud = Pointclouds(points=[torch.tensor(points, dtype=torch.float32)])

        sampled_points, sampled_indices = sample_farthest_points(L0_point_cloud.points_padded(), K=1024)


        # print(" sampled_points shape is ", sampled_points.shape)
        # print(" sampled_indices shape is ", sampled_indices.shape)


        sampled_points = sampled_points[0].numpy()
        sampled_indices = sampled_indices[0].numpy()

         # pc中心化
        bbox_center = (sampled_points.min(axis=0) + sampled_points.max(axis=0)) / 2
        centralized_point_cloud = sampled_points - bbox_center

        sampled_L0_part_indices = self.get_sampled_part_indices(sampled_indices, L0_point_indices)
        sampled_L1_part_indices = self.get_sampled_part_indices(sampled_indices, L1_point_indices)

        sampled_L0_part_indices = self.get_sampled_part_indices(sampled_indices, L0_point_indices)
        sampled_L1_part_indices = self.get_sampled_part_indices(sampled_indices, L1_point_indices)

        L0_part_mask = np.zeros((3, 4096), dtype=np.float32)

        L1_part_mask = np.zeros((6, 4096), dtype=np.float32)

        L0_index = 0
        for part_idx, point_indices in sampled_L0_part_indices.items():
            print(f"part_idx is {part_idx}, len is {len(point_indices)}")
            for point_idx in point_indices:
                L0_part_mask[L0_index, point_idx] = 1
            L0_index = L0_index + 1
        L1_index = 0
        for part_idx, point_indices in sampled_L1_part_indices.items():
            print(f"part_idx is {part_idx}, len is {len(point_indices)}")
            for point_idx in point_indices:
                L1_part_mask[L1_index, point_idx] = 1
            L1_index = L1_index +1
        
        # for i in range(L0_index):
        #     print(f"L0_part_mask[{i}], sum is {sum(L0_part_mask[i])}")

        # for i in range(L1_index):
        #     print(f"L1_part_mask[{i}], sum is {sum(L1_part_mask[i])}")

        return centralized_point_cloud, L0_part_mask, L1_part_mask
    
    def get_pointcloud_w_part_load(self, idx):
        cate_id = self.grasp_list[idx]["cate_id"]      # 物体类型
        obj_id = self.grasp_list[idx]["obj_id"]        # 物体id
        act_id = self.grasp_list[idx]["action_id"] 
        intent_name = self.action_id_to_intent[act_id] # 意图
        # pc = np.asarray(self.base_dataset.get_obj_pc_path(idx)).astype(np.int32)
        raw_pc_path, obj_name = self.base_dataset.get_obj_pc_path(idx)  # 带RGB的点云
        

        base_path = "/home/syks/Scene-Diffuser-obj/data/OakInk/shape/preprocess"
        save_path = os.path.join(base_path, obj_name, "preprocess.npz")
        data = np.load(save_path)
        centralized_point_cloud = data["centralized_point_cloud"]
        L0_part_mask = data["L0_part_mask"]
        L1_part_mask = data["L1_part_mask"]

        return centralized_point_cloud, L0_part_mask, L1_part_mask
    

    def get_pointcloud_SV(self, idx, sv_id):
        cate_id = self.grasp_list[idx]["cate_id"]      # 物体类型
        

        obj_id = self.grasp_list[idx]["obj_id"]        # 物体id
        obj_name, real_flag = self.base_dataset.get_obj_name(idx)
        # SV_id = np.random.randint(0, 10)
        # 根据物体id获取物体name
        # model_align_  这个有问题


        # if real_flag:
        #     pc_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/render/OakInkObjectsV2/pcd/{obj_name}/align_ds/model_align_{sv_id}.ply"
        # else:
        #     pc_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/render/OakInkVirtualObjectsV2/pcd/{obj_name}/align_ds/model_align_{sv_id}.ply"

        if real_flag:
            pc_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/render/OakInkObjectsV2/pcd/{obj_name}/align_ds/"
        else:
            pc_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/render/OakInkVirtualObjectsV2/pcd/{obj_name}/align_ds/"
        all_files = os.listdir(pc_path)
        pc_files = [f for f in all_files if f.endswith(f'_{sv_id}.ply')]
        pc_path = os.path.join(pc_path, pc_files[0])

        # print(f"sv_id is  {sv_id}, the pc_path is {pc_path}")
        
        points, _ = self.read_pointcloud(pc_path)

        L0_point_cloud = Pointclouds(points=[torch.tensor(points, dtype=torch.float32)])

        sampled_points, sampled_indices = sample_farthest_points(L0_point_cloud.points_padded(), K=1024)


        # print(" sampled_points shape is ", sampled_points.shape)
        # print(" sampled_indices shape is ", sampled_indices.shape)


        sampled_points = sampled_points[0].numpy()
        sampled_indices = sampled_indices[0].numpy()

         # pc中心化
        bbox_center = (sampled_points.min(axis=0) + sampled_points.max(axis=0)) / 2
        centralized_point_cloud = sampled_points - bbox_center

        return centralized_point_cloud




   

        # 把4096个点云放进来，还有就是mask（最多12个），part无论怎么样，输入到pointnet的输出维度都是一样的

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
    
    


import re

def parse_cate_and_obj_id(output_string):
    # 定义正则表达式模式，只捕获 cate_id 和 obj_id
    pattern = re.compile(
        r"_CATE_(?P<cate_id>.+?)"  # 非贪婪匹配 cate_id
        r"_OBJ\((?P<obj_id>.+?)\)"  # 非贪婪匹配 obj_id，到第一个右括号为止
    )
    
    # 使用正则表达式匹配并提取参数
    match = pattern.search(output_string)
    if not match:
        raise ValueError("输出字符串格式不正确")
    
    # 返回提取出的参数
    return {
        'cate_id': match.group('cate_id'),
        'obj_id': match.group('obj_id')
    }




x_lower = torch.tensor([-2.8087e+00, -2.7107e+00, -3.0719e+00, -2.7644e-01, -8.1655e-01,
         2.4890e-04, -4.2917e-01, -2.2007e-01, -2.6136e-03, -6.7771e-02,
        -1.9199e-01,  1.1939e-04, -6.2365e-01, -7.9654e-01,  1.4647e-02,
        -4.1412e-01, -5.8028e-01,  1.8073e-04, -5.1573e-01, -1.9574e-01,
         1.2893e-04, -1.1803e+00, -6.7920e-01,  5.8778e-03, -1.0473e+00,
        -4.1329e-01,  7.6797e-05, -1.0065e+00, -4.8045e-01,  1.3998e-04,
        -6.3780e-01, -7.0535e-01,  7.0273e-03, -7.4499e-01, -4.0315e-01,
         1.4356e-04, -7.7549e-01, -2.4576e-01,  1.0434e-04, -5.6672e-01,
        -1.2392e+00, -2.6292e-01, -2.9719e-01, -7.6436e-01, -3.0848e-01,
        -5.3309e-02, -1.1928e+00, -1.1817e-01, -4.7334e-02, -3.4845e-02,
        -1.3147e-02, -2.1503e-02, -3.5874e-02, -2.6014e-02, -1.4266e-02,
        -8.3219e-03, -2.1315e-02, -2.7304e-02, -1.4562e-01, -1.1820e-01,
        -1.4180e-01])
x_upper = torch.tensor([ 2.8045e+00,  2.8593e+00,  3.0458e+00,  3.7927e-01,  5.2747e-01,
         1.6540e+00,  1.9258e-01,  3.0968e-01,  1.6662e+00,  5.0568e-01,
         2.6016e-01,  1.6478e+00,  1.4105e-01,  7.8212e-01,  1.7043e+00,
         1.3340e-01,  2.0309e-01,  1.6089e+00,  1.6088e-02,  2.0297e-01,
         1.7031e+00, -2.8203e-03,  8.9844e-01,  1.4268e+00,  7.4993e-02,
         2.0151e-01,  1.4444e+00, -9.0373e-05,  1.7854e-01,  1.4866e+00, 
         1.5842e-01,  7.0282e-01,  1.6982e+00,  7.9677e-02,  3.2444e-01,
         1.6026e+00, -3.1889e-05,  2.2176e-01,  1.6037e+00,  1.4480e+00,
         4.9252e-01,  1.1986e+00,  8.1752e-01,  5.8521e-01,  1.0878e+00,
         6.4195e-01,  1.0558e-01,  6.0882e-01,  3.8521e-02,  1.3721e-02,
         2.5672e-02,  3.5441e-02,  2.0428e-02,  4.1258e-02,  1.9818e-02,
         1.3290e-02,  2.0268e-02,  1.8080e-02,  1.4512e-01,  1.5682e-01,
         1.7509e-01])

_NORMALIZE_LOWER = -1.
_NORMALIZE_UPPER = 1.

def trans_denormalize(global_trans: torch.Tensor):
    global_trans_denorm = global_trans + (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
    global_trans_denorm /= (_NORMALIZE_UPPER - _NORMALIZE_LOWER)
    global_trans_denorm = global_trans_denorm * (x_upper - x_lower) + x_lower
    return global_trans_denorm

from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch

from datasets.viz_o3d_utils import VizContext

from matplotlib.colors import get_named_colors_mapping

Cmap = get_named_colors_mapping()


def main(args):
    # OISHAPE_CONFIG = dict(
    #     DATA_SPLIT=args.data_split,
    #     DATA_ROOT="data",
    #     OBJ_CATES=args.categories,
    #     INTENT_MODE=args.intent_mode,
    #     DATA_PRESET=dict(
    #         CENTER_IDX=9,
    #         USE_CACHE=True,
    #         N_RESAMPLED_OBJ_POINTS=4096,
    #     ),
    # )
    # cfg = CN(OISHAPE_CONFIG)
    config_path = "configs/task/AnyDexTOG_test.yaml"
    cfg = OmegaConf.load(config_path)
    dataset: OIShape = OIShape(cfg, "train",False)
    dataloader = dataset.get_dataloader(batch_size=1,
                                                                                  collate_fn=collate_fn_general,
                                                                                  num_workers=0,
                                                                                  pin_memory=False,
                                                                                  shuffle=True,)
    data_iter = iter(dataset)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
    viz_context = VizContext()
    viz_context.init(point_size=10.0)
    # mano_layer = ManoLayer(center_idx=dataset.center_idx, mano_assets_root="assets/mano_v1_2")
    mano_layer = ManoLayer(
            rot_mode="axisang",
            center_idx=9,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )
    hand_faces_new = mano_layer.get_mano_closed_faces().numpy()

    def next_sample(_):
        grasp_item = next(data_iter)
        # grasp_item = transform(grasp_item)
        sample_id = grasp_item[Queries.SAMPLE_IDENTIFIER]
        print(sample_id[0])
        # if sample_id not in ["scissors", "lotion_pump", "squeezable", "wrench", "power_drill"]:
        #     grasp_item = next(data_iter)
        # print("grasp_item is ", grasp_item)
        # for key in grasp_item.keys():
        #     print("--------------------------------------------------------------------------------------")
        #     print(f"key is {key}")
        #     print("is ", grasp_item[key])
        #     try:
        #         print("shape is , ", grasp_item[key].shape)
        #     except:
        #         print("not shape ,len is",grasp_item[key])
        #     print("--------------------------------------------------------------------------------------")

        real_meta = json.load(open(os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2", "object_id.json"), "r"))
        virtual_meta = json.load(open(os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2", "virtual_object_id.json"), "r"))
        oid = grasp_item[Queries.OBJ_ID]
        # print(" oid is ", oid)
        if oid in real_meta:
            obj_name = real_meta[oid]["name"]
            
        else:
            obj_name = virtual_meta[oid]["name"]
            
        # obj_mesh_path = list(
        #     glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
        #     glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply")))

        pc_path = os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape", "ply_new")
        obj_pc_path = os.path.join(pc_path, obj_name, "align", "output_point_cloud.ply")
        pcd = o3d.io.read_point_cloud(obj_pc_path)
        points = np.asarray(pcd.points)
        print("points shape is ", points.shape)
        # dense_pc = np.load(obj_pc_path, allow_pickle= True)



        
        parsed_params = parse_cate_and_obj_id(grasp_item['sample_identifier'])
        print(" parsed_params is ", parsed_params["cate_id"])
        if Queries.HAND_VERTS_OBJ in grasp_item:
            hand_verts = grasp_item[Queries.HAND_VERTS_OBJ].squeeze(0).numpy()
            hand_faces = grasp_item[Queries.HAND_FACES].squeeze(0).numpy()
            hand_verts_f = grasp_item[f"{Queries.HAND_VERTS_OBJ}_f"].squeeze(0).numpy()
            viz_context.update_by_mesh("hand_f", hand_verts_f, hand_faces, vcolors=Cmap["deepskyblue"], update=True)
            viz_context.update_by_mesh("hand", hand_verts, hand_faces, vcolors=Cmap["tomato"], update=True)


        hand_pose = grasp_item[Queries.HAND_POSE_OBJ]
        
        hand_shape = grasp_item[Queries.HAND_SHAPE]
        mano_output = mano_layer(torch.from_numpy(hand_pose).unsqueeze(0).float(),torch.from_numpy(hand_shape).unsqueeze(0).float()).verts.squeeze(0).numpy()
        # print("grasp_item[Queries.JOINTS_OBJ] is ", grasp_item[Queries.JOINTS_OBJ].shape)

        hand_transl_obj = grasp_item[Queries.JOINTS_OBJ][9,:]
        # print("hand_transl_obj is ", hand_transl_obj.shape)
        hand_verts_obj = mano_output+ hand_transl_obj


        viz_context.update_by_mesh("hand", hand_verts_obj, hand_faces_new, vcolors=Cmap["tomato"], update=True)
        # print("grasp_item[Queries.OBJ_VERTS_OBJ_DS_ORI] shape is ", grasp_item[Queries.OBJ_VERTS_OBJ_DS_ORI].shape)
        obj_verts_ds = grasp_item[Queries.OBJ_VERTS_OBJ_DS_ORI]
        obj_normals_ds = grasp_item[Queries.OBJ_NORMALS_OBJ_DS]
        # print(" obj_verts_ds is ", obj_verts_ds, "shape obj_verts_ds is ", obj_verts_ds.shape)
        viz_context.update_by_pc("obj_ori", obj_verts_ds, obj_normals_ds, pcolors=Cmap["black"], update=True)
        obj_verts_ds = grasp_item[Queries.OBJ_VERTS_OBJ_DS]

        print(" obj_verts_ds is ", obj_verts_ds, "shape obj_verts_ds is ", obj_verts_ds.shape)
        # viz_context.update_by_pc("obj", points, obj_normals_ds, pcolors=Cmap["deepskyblue"], update=True)
        viz_context.update_by_pc("obj", obj_verts_ds.squeeze(0), obj_normals_ds, pcolors=Cmap["red"], update=True)


        # viz_context.update_by_pc("obj_dense", points, obj_normals_ds, pcolors=Cmap["deepskyblue"], update=True)

        

    next_sample(viz_context)
    viz_context.register_key_callback("D", next_sample)
    viz_context.run()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # mano_layer = ManoLayer(center_idx=dataset.center_idx, mano_assets_root="assets/mano_v1_2")
    # hand_faces = mano_layer.get_mano_closed_faces().numpy()
    # max_values = torch.full((61,), float('-inf'))  # 初始为负无穷大
    # min_values = torch.full((61,), float('inf'))   # 初始为正无穷大


    for it, data in enumerate(dataloader):
    #     for key in data:
    #         if torch.is_tensor(data[key]):
    #             data[key] = data[key]
    #     print("data[x] shape is ", data["x"][0].shape)
    #     labels = data["x"]
    #     for label in labels:
    #         if torch.is_tensor(label):
    #             # 更新最大值和最小值
    #             max_values = torch.max(max_values, label)
    #             min_values = torch.min(min_values, label)

                
    #         else:
    #             print("skip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # print("Max values for each dimension:", max_values)
    # print("Min values for each dimension:", min_values)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # print(data.keys())
        # print(data['sample_identifier'][0])
        parsed_params = parse_cate_and_obj_id(data['sample_identifier'][0])
        # print("data obj_rotmat is ", data['obj_rotmat'])
        parsed_params = parse_cate_and_obj_id(data['sample_identifier'][0])
        # print(" parsed_params is ", parsed_params["cate_id"])
        # print(f"------------------------------------------------------------------------------------------------------------------")
        # print("data obj_part_name is ", data['obj_valid_part'], "data['obj_part_name'] type is ", type(data['obj_valid_part']))
        # obj_valid_part = torch.tensor(data['obj_valid_part'])
        # print("obj_part_name shape is ", obj_valid_part.shape)
        # print("data obj_l0_part is ", data['obj_l0_part'], "type is ", type(data['obj_l0_part']))

        # for index, part in enumerate(data['obj_l0_part']):

        #     print(f"part {part} len is {len(data['obj_l0_part'][index])}")

        # print(data['x'].shape)
        real_meta = json.load(open(os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2", "object_id.json"), "r"))
        virtual_meta = json.load(open(os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2", "virtual_object_id.json"), "r"))

        # print(" oid is ", data['obj_id'])
        if data['obj_id'][0] in real_meta:
            obj_name = real_meta[data['obj_id'][0]]["name"]
            
        else:
            obj_name = virtual_meta[data['obj_id'][0]]["name"]

        # print("obj_name is ", obj_name)
        # print("intent_name is  ", data['intent_name'])
        print(f"{obj_name} {data['intent_name'][0]} {parsed_params['cate_id']} {data['obj_id'][0]}")
        # print(f"{obj_name}")
        # print("data[ins_disc] shape is,", data["ins_disc"].shape)
        # print("data[ins_mask_disc] shape is", {data["ins_mask_disc"].shape})
        # print("data[task_disc] shape is", data["task_disc"].shape)
        # print("data[task_mask_disc] shape is", data["task_mask_disc"].shape)
        # print("data[obj_part_disc] shape is", data["obj_part_disc"].shape)
        # print("data[obj_part_disc_mask] shape is", data["obj_part_disc_mask"].shape)
        # print("data[obj_valid_part] shape is", data["obj_valid_part"].shape)


        
        # print("obj_l0_part is ", data["obj_l0_part"])
        # print("obj_l0_part is ", data["obj_l1_part"])
        # print("-------------------------------------------------------------------------------------------------------------------")
    # def next_sample(_):
    #     grasp_item = next(dataloader)
    #     hand_pose_obj = grasp_item[Queries.HAND_POSE_OBJ]
    #     hand_shape = grasp_item[Queries.HAND_SHAPE]
        # hand_verts = mano_layer(
        #     torch.from_numpy(hand_pose_obj).unsqueeze(0),
        #     torch.from_numpy(hand_shape).unsqueeze(0)).verts.squeeze(0).numpy()

    #     joint_obj = grasp_item[Queries.JOINTS_OBJ]
    #     root_joint_obj = joint_obj[dataset.center_idx, :]
    #     hand_verts_obj = hand_verts + root_joint_obj
        

    #     obj_verts_ds = grasp_item[Queries.OBJ_VERTS_OBJ_DS]
    #     obj_normals_ds = grasp_item[Queries.OBJ_NORMALS_OBJ_DS]


    # next_sample(None)



# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description="viz grabnet grasp")
#     parser.add_argument("--data_dir", type=str, default="data/OakInk", help="environment variable 'OAKINK_DIR'")
#     parser.add_argument("--categories", type=str, default="val", help="list of object categories")
#     parser.add_argument("--intent_mode",
#                         type=list,
#                         action="append",
#                         default=["use", "hold", "liftup"],
#                         choices=["use", "hold", "liftup", "handover"],
#                         help="intent mode, list of intents")
#     parser.add_argument("--data_split",
#                         type=str,
#                         default="all",
#                         choices=["train", "test", "val", "all"],
#                         help="data split")

#     args = parser.parse_args()
#     main(args)

def read_pointcloud( ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    colors = np.round(colors, 3)
    return points, colors

if __name__ == '__main__':

    config_path = "configs/task/SV_test.yaml"
    cfg = OmegaConf.load(config_path)
    dataset: OIShape_SV = OIShape_SV(cfg, "all",False)
    dataloader = dataset.get_dataloader(batch_size=1,
                                                                                  collate_fn=collate_fn_general,
                                                                                  num_workers=0,
                                                                                  pin_memory=False,
                                                                                  shuffle=True,)
    data_iter = iter(dataset)
    viz_context = VizContext()
    viz_context.init(point_size=10.0)
    # mano_layer = ManoLayer(center_idx=dataset.center_idx, mano_assets_root="assets/mano_v1_2")
    mano_layer = ManoLayer(
            rot_mode="axisang",
            center_idx=9,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )
    hand_faces_new = mano_layer.get_mano_closed_faces().numpy()

    def next_sample(_):
        grasp_item = next(data_iter)
        # grasp_item = transform(grasp_item)
        sample_id = grasp_item[Queries.SAMPLE_IDENTIFIER]
        print(sample_id[0])
        # if sample_id not in ["scissors", "lotion_pump", "squeezable", "wrench", "power_drill"]:
        #     grasp_item = next(data_iter)
        # print("grasp_item is ", grasp_item)
        # for key in grasp_item.keys():
        #     print("--------------------------------------------------------------------------------------")
        #     print(f"key is {key}")
        #     print("is ", grasp_item[key])
        #     try:
        #         print("shape is , ", grasp_item[key].shape)
        #     except:
        #         print("not shape ,len is",grasp_item[key])
        #     print("--------------------------------------------------------------------------------------")

        real_meta = json.load(open(os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2", "object_id.json"), "r"))
        virtual_meta = json.load(open(os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2", "virtual_object_id.json"), "r"))
        oid = grasp_item[Queries.OBJ_ID]
        # print(" oid is ", oid)
        real_flag = False
        if oid in real_meta:
            obj_name = real_meta[oid]["name"]
            real_flag = True
            
        else:
            obj_name = virtual_meta[oid]["name"]
            
        # obj_mesh_path = list(
        #     glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
        #     glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply")))
        if real_flag:
            pc_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/render/OakInkObjectsV2/pcd/{obj_name}/align_ds/"
        else:
            pc_path = f"/home/syks/Scene-Diffuser-obj/data/OakInk/render/OakInkVirtualObjectsV2/pcd/{obj_name}/align_ds/"

        sv_id = np.random.randint(0, 10)

        all_files = os.listdir(pc_path)
        pc_files = [f for f in all_files if f.endswith(f'_{sv_id}.ply')]
        pc_path = os.path.join(pc_path, pc_files[0])

        points, _ = read_pointcloud(pc_path)
        L0_point_cloud = Pointclouds(points=[torch.tensor(points, dtype=torch.float32)])

        sampled_points, sampled_indices = sample_farthest_points(L0_point_cloud.points_padded(), K=1024)

        sampled_points = sampled_points[0].numpy()
        sampled_indices = sampled_indices[0].numpy()
        bbox_center = (sampled_points.min(axis=0) + sampled_points.max(axis=0)) / 2
        centralized_point_cloud = sampled_points - bbox_center



        pcd = o3d.io.read_point_cloud(pc_path)
        points = np.asarray(pcd.points)
        print("points shape is ", points.shape)
        # dense_pc = np.load(obj_pc_path, allow_pickle= True)



        
        parsed_params = parse_cate_and_obj_id(grasp_item['sample_identifier'])
        print(" parsed_params is ", parsed_params["cate_id"])
        if Queries.HAND_VERTS_OBJ in grasp_item:
            hand_verts = grasp_item[Queries.HAND_VERTS_OBJ].squeeze(0).numpy()
            hand_faces = grasp_item[Queries.HAND_FACES].squeeze(0).numpy()
            hand_verts_f = grasp_item[f"{Queries.HAND_VERTS_OBJ}_f"].squeeze(0).numpy()
            viz_context.update_by_mesh("hand_f", hand_verts_f, hand_faces, vcolors=Cmap["deepskyblue"], update=True)
            viz_context.update_by_mesh("hand", hand_verts, hand_faces, vcolors=Cmap["tomato"], update=True)


        hand_pose = grasp_item[Queries.HAND_POSE_OBJ]
        
        hand_shape = grasp_item[Queries.HAND_SHAPE]
        mano_output = mano_layer(torch.from_numpy(hand_pose).unsqueeze(0).float(),torch.from_numpy(hand_shape).unsqueeze(0).float()).verts.squeeze(0).numpy()
        # print("grasp_item[Queries.JOINTS_OBJ] is ", grasp_item[Queries.JOINTS_OBJ].shape)

        hand_transl_obj = grasp_item[Queries.JOINTS_OBJ][9,:]
        # print("hand_transl_obj is ", hand_transl_obj.shape)
        hand_verts_obj = mano_output+ hand_transl_obj


        # viz_context.update_by_mesh("hand", hand_verts_obj, hand_faces_new, vcolors=Cmap["tomato"], update=True)
        # print("grasp_item[Queries.OBJ_VERTS_OBJ_DS_ORI] shape is ", grasp_item[Queries.OBJ_VERTS_OBJ_DS_ORI].shape)
        obj_verts_ds = grasp_item[Queries.OBJ_VERTS_OBJ_DS_ORI]
        obj_normals_ds = grasp_item[Queries.OBJ_NORMALS_OBJ_DS]
        # print(" obj_verts_ds is ", obj_verts_ds, "shape obj_verts_ds is ", obj_verts_ds.shape)
        viz_context.update_by_pc("obj_ori", obj_verts_ds, obj_normals_ds, pcolors=Cmap["green"], update=True)
        obj_verts_ds = grasp_item[Queries.OBJ_VERTS_OBJ_DS]

        print(" obj_verts_ds is ", obj_verts_ds, "shape obj_verts_ds is ", obj_verts_ds.shape)
        # viz_context.update_by_pc("obj", points, obj_normals_ds, pcolors=Cmap["deepskyblue"], update=True)
        viz_context.update_by_pc("obj", obj_verts_ds.squeeze(0), obj_normals_ds, pcolors=Cmap["red"], update=True)


        viz_context.update_by_pc("obj_dense", centralized_point_cloud, obj_normals_ds, pcolors=Cmap["deepskyblue"], update=True)

        

    next_sample(viz_context)
    viz_context.register_key_callback("D", next_sample)
    viz_context.run()

# -------------------------------------------------------------------------------------------------------------------------------------------------
# 求最大最小值
#     global_min = None
#     global_max = None

#     device = 'cuda'
#     for it, data in enumerate(dataloader):
#         # parsed_params = parse_cate_and_obj_id(data['sample_identifier'][0])
#         # # print("data obj_rotmat is ", data['obj_rotmat'])
#         # parsed_params = parse_cate_and_obj_id(data['sample_identifier'][0])
#         # # print(" parsed_params is ", parsed_params["cate_id"])
#         # # print(f"------------------------------------------------------------------------------------------------------------------")
#         # # print("data obj_part_name is ", data['obj_valid_part'], "data['obj_part_name'] type is ", type(data['obj_valid_part']))
#         # # obj_valid_part = torch.tensor(data['obj_valid_part'])
#         # # print("obj_part_name shape is ", obj_valid_part.shape)
#         # # print("data obj_l0_part is ", data['obj_l0_part'], "type is ", type(data['obj_l0_part']))

#         # # for index, part in enumerate(data['obj_l0_part']):

#         # #     print(f"part {part} len is {len(data['obj_l0_part'][index])}")

#         # # print(data['x'].shape)
#         # real_meta = json.load(open(os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2", "object_id.json"), "r"))
#         # virtual_meta = json.load(open(os.path.join("/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2", "virtual_object_id.json"), "r"))

#         # # print(" oid is ", data['obj_id'])
#         # if data['obj_id'][0] in real_meta:
#         #     obj_name = real_meta[data['obj_id'][0]]["name"]
            
#         # else:
#         #     obj_name = virtual_meta[data['obj_id'][0]]["name"]

#         # # print("obj_name is ", obj_name)
#         # # print("intent_name is  ", data['intent_name'])
#         # print(f"{obj_name} {data['ins_path'][0]} {parsed_params['cate_id']} {data['obj_id'][0]}")


#         for key in data:
#             if torch.is_tensor(data[key]):
#                 data[key] = data[key].to(device)
            
#             if key == 'x':
#                 print("x shape is ", data[key].shape)
#                 x_batch = data[key]
#                 batch_min = torch.min(x_batch, dim=0).values
#                 batch_max = torch.max(x_batch, dim=0).values
#                     # 记录x中每一维度的最大值和最小值
#                 if global_min is None and global_max is None:
#                         global_min = batch_min
#                         global_max = batch_max
#                 else:
#                     # 更新全局最小值和最大值
#                     global_min = torch.min(global_min, batch_min)
#                     global_max = torch.max(global_max, batch_max)

#                     print(data[key].shape)
#             # print("global_min is ", global_min)
#             # print("global_max is ", global_max)
#         print("next one !!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print("global_min is ", global_min)
#     print("global_max is ", global_max)
# -------------------------------------------------------------------------------------------------------------------------------------------------
