import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
import pickle
from omegaconf import DictConfig
from plotly import graph_objects as go
from typing import Any
from scipy.spatial import cKDTree
from utils.misc import random_str
from utils.registry import Registry
from utils.visualize import frame2gif, render_prox_scene, render_scannet_path
from utils.visualize import create_trimesh_nodes_path, create_trimesh_node
from utils.mano_graspit_handmodel import get_handmodel  
from utils.plotly_utils import plot_mesh
from utils.rot6d import rot_to_orthod6d, robust_compute_rotation_matrix_from_ortho6d, random_rot
from tqdm import tqdm
import json
import glob
from manotorch.manolayer import ManoLayer
import open3d as o3d

from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.knn import knn_gather, knn_points
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

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
    "hammers",
    "screwdriver",
    "fryingpan",
    "teapot",
    "game_controller"
]

VISUALIZER = Registry('Visualizer')

import re
import json
def parse_cate_and_obj_id(output_string):

    pattern = re.compile(
        r"_CATE_(?P<cate_id>.+?)"  
        r"_OBJ\((?P<obj_id>.+?)\)"  
    )
    match = pattern.search(output_string)
    if not match:
        raise ValueError("输出字符串格式不正确")
    
    return {
        'cate_id': match.group('cate_id'),
        'obj_id': match.group('obj_id')
    }

def parse_all(output_string):
    pattern = re.compile(
        r"_CATE_(?P<cate_id>.+?)"  
        r"_OBJ\((?P<obj_id>.+?)\)" 
        r"_INT\((?P<int_name>.+?)\)"
    )
    match = pattern.search(output_string)
    if not match:
        raise ValueError("输出字符串格式不正确")
    
    return {
        'cate_id': match.group('cate_id'),
        'obj_id': match.group('obj_id'),
        'int_name': match.group('int_name')
    }

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        #self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        #trans = self.stn(x)
        x = x.transpose(2, 1)  # [B, N, D]
        if D > 3 :
            x, feature = x.split(3,dim=2)
        #x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))  # [B, N, 64]

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # global feature: [B, 1024]
        if self.global_feat:
            return x, None, trans_feat
            #return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)  # N  [B, 1024, N]
            return torch.cat([x, pointfeat], 1), None, trans_feat
            #return torch.cat([x, pointfeat], 1), trans, trans_feat


class pointnet_reg(nn.Module):
    def __init__(self, num_class=1, with_rgb=True):
        super(pointnet_reg, self).__init__()
        if with_rgb:
            channel = 6
        else:
            channel = 3
        self.k = num_class
        self.feat_o = PointNetEncoder(global_feat=False, feature_transform=False, channel=channel)  # feature trans True
        self.feat_h = PointNetEncoder(global_feat=False, feature_transform=False, channel=channel)  # feature trans True
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.convfuse = torch.nn.Conv1d(3778, 3000, 1)
        self.bnfuse = nn.BatchNorm1d(3000)

    def forward(self, x, hand):
        '''
        :param x: obj pc [B, D, N]
        :param hand: hand pc [B, D, 778]
        :return: regressed cmap
        '''
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # for obj
        x, trans, trans_feat = self.feat_o(x)  # x: [B, 1088, N] global+point feature of object
        # for hand
        hand, trans2, trans_feat2 = self.feat_h(hand)  # hand: [B, 1088, 778] global+point feature of hand
        # fuse feature of object and hand
        x = torch.cat((x, hand), dim=2).permute(0,2,1).contiguous()  # [B, N+778, 1088]
        x = F.relu(self.bnfuse(self.convfuse(x)))  # [B, N, 1088]
        x = x.permute(0,2,1).contiguous()  # [B, 1088, N]
        # inference cmap
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N]
        x = self.conv4(x)  # [B, 1, N]
        x = x.transpose(2,1).contiguous()
        x = torch.sigmoid(x)
        x = x.view(batchsize, n_pts)  # n_pts  [B, N]
        return x

def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    '''
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx

def get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    '''
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    '''
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

    interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
    return interior

def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)

from pytorch3d.structures import Meshes

def Contact_loss(obj_xyz, hand_xyz, cmap):
    '''
    # hand-centric loss, encouraging hand touching object surface
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N1], dynamic possible contact regions on object
    :param hand_faces_index: [B, 1538, 3] hand index in [0, N2-1]
    :return:
    '''
    f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
          750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
    f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
          327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
          355]
    f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
          439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
    f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
          550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
    f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
          668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
    f0 = [73, 96, 98, 99, 772, 774, 775, 777]
    prior_idx = f1 + f2 + f3 + f4 + f5 + f0
    hand_xyz_prior = hand_xyz[:, prior_idx, :]  # only using prior points for contact map

    B = obj_xyz.size(0)

    obj_CD, _ = get_NN(obj_xyz, hand_xyz_prior)  # [B, N1] NN distance from obj pc to hand pc
    n_points = torch.sum(cmap)
    cmap_loss = obj_CD[cmap].sum() / (B * n_points)
    return 3000.0 * cmap_loss

def get_pseudo_cmap(nn_dists):
    '''
    calculate pseudo contactmap: 0~3cm mapped into value 1~0
    :param nn_dists: object nn distance [B, N] or [N,] in meter**2
    :return: pseudo contactmap [B,N] or [N,] range in [0,1]
    '''
    nn_dists = 100.0 * torch.sqrt(nn_dists)  # turn into center-meter
    cmap = 1.0 - 2 * (torch.sigmoid(nn_dists*2) -0.5)
    return cmap

def TTT_loss(hand_xyz, hand_face, obj_xyz, cmap_affordance, cmap_pointnet):
    '''
    :param hand_xyz:
    :param hand_face:
    :param obj_xyz:
    :param cmap_affordance: contact map calculated from predicted hand mesh
    :param cmap_pointnet: target contact map predicted from ContactNet
    :return:
    '''
    B = hand_xyz.size(0)
    # print("hand_xyz shape is ", hand_xyz.shape , "hand_face shape is ", hand_face.shape)
    # hand_face = torch.tensor(hand_face)
    hand_face_tensor = torch.from_numpy(hand_face).float()
    hand_face_tensor = hand_face_tensor.unsqueeze(0).expand(B, -1, -1)
    # inter-penetration loss
    mesh = Meshes(verts=hand_xyz.cuda(), faces=hand_face_tensor.cuda())
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
    nn_dist, nn_idx = get_NN(obj_xyz, hand_xyz)
    interior = get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)
    penetr_dist = 120 * nn_dist[interior].sum() / B  # batch reduction

    # cmap consistency loss
    consistency_loss = 0.0001 * torch.nn.functional.mse_loss(cmap_affordance, cmap_pointnet, reduction='none').sum() / B
    
    # hand-centric loss
    contact_loss = 2.5 * Contact_loss(obj_xyz, hand_xyz, cmap=nn_dist < 0.02**2)
    return penetr_dist, consistency_loss, contact_loss

@VISUALIZER.register()
@torch.no_grad()
class PPTVisualizer():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.
        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample

    

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

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self.x_upper - self.x_lower) + self.x_lower
        return global_trans_denorm
    

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def read_pointcloud(self, ply_path):
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        colors = np.round(colors, 3)
        return points, colors
      
    def visualize(
            self,
            model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            save_dir: str
    ) -> None:
        """ Visualize method
        Args:
            model: diffusion model
            dataloader: test dataloader
            save_dir: save directory of rendering images
        """
        model.eval()
        device = model.device
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'html'), exist_ok=True)


        mano_layer = ManoLayer(
            rot_mode="axisang",
            center_idx=9,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )
        hand_faces = mano_layer.get_mano_closed_faces().numpy()


        
        test_split_dir = os.path.join("/home/syks/Partial-Point-cloud-TOG-main/txt/test.txt") 
        complete_file_path = '/home/syks/Scene-Diffuser-obj/data/OakInk/complete_test'
        
        test_split = []
        with open(test_split_dir, 'r') as file:
            lines = file.readlines()
            for line in lines:
                test_split.append(line.replace("\n", ""))

        pbar = tqdm(total=len(test_split) * self.ksample)

        res = {'method': 'TOSC',
               'desc': 'TOSC',
               'sample_qpos': {}, 
               'sample_part':{},
               'select_obj':{},
               'select_part':{}
               }

        for object in test_split:
            file_path = f"{complete_file_path}/{object}//viewpoint.json"
            with open(file_path, 'r') as f:
                json_file = json.load(f)
            task = json_file[-1]['guidence']
            sample_identifier = json_file[-1]['sample_identifier']
            viewpoint = json_file[-1]['position']
            parsed_params = parse_all(sample_identifier)
            oid = parsed_params["obj_id"]
            object_name = parsed_params["cate_id"]
            int_name = parsed_params["int_name"]

            pc_path = os.path.join(complete_file_path, object, "complete.npy")
            sampled_points = np.load(pc_path)
            bbox_center = (sampled_points.min(axis=0) + sampled_points.max(axis=0)) / 2
            centralized_point_cloud = sampled_points - bbox_center
            obj_pcd_can = torch.tensor(centralized_point_cloud, device=device).unsqueeze(0).repeat(self.ksample, 1, 1)
            partial_pc_path = os.path.join(complete_file_path, object, "partial_point_cloud.ply") 
            partial_sampled_points, _ = self.read_pointcloud(partial_pc_path)
            partial_bbox_center = (partial_sampled_points.min(axis=0) + partial_sampled_points.max(axis=0)) / 2
            partial_centralized_point_cloud = partial_sampled_points - partial_bbox_center
            partial_obj_pcd_can = torch.tensor(partial_centralized_point_cloud, device=device).unsqueeze(0).repeat(self.ksample, 1, 1)
            guidence_list = []

            for i in range(self.ksample):
                guidence_list.append(task)
            data = {'x': torch.randn(self.ksample, 61, device=device),  
                    'pos': obj_pcd_can.to(device),
                    'pos_ori': partial_obj_pcd_can.to(device),
                    'guidence': guidence_list,
                    
                    }
            
            for key in data.keys():
                if key in ['x', 'pos', 'feat', 's_grid_sdf', 's_grid_min', 's_grid_max', 'start', 'end', 'ins_mask_disc', 'task_disc', 'task_mask_disc','obj_L0_part_disc', 'obj_L0_part_disc_mask', 'obj_L0_valid_part','obj_L1_part_disc', 'obj_L1_part_disc_mask', 'obj_L1_valid_part'] and not torch.is_tensor(data[key]):
                    data[key] = torch.tensor(np.array(data[key]))
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            
            outputs = model.sample(data, k=1).squeeze(1).to(torch.float64) # flow-matching的
            obj_suffix_path = "align_ds"
            data_path = "/home/syks/Scene-Diffuser-obj/data/OakInk/shape"
            meta_path = "/home/syks/Scene-Diffuser-obj/data/OakInk/shape/metaV2"
            real_meta = json.load(open(os.path.join(meta_path, "object_id.json"), "r"))
            virtual_meta = json.load(open(os.path.join(meta_path, "virtual_object_id.json"), "r"))

            if oid in real_meta:
                obj_name = real_meta[oid]["name"]
                obj_path = os.path.join(data_path, "OakInkObjectsV2")
            else:
                obj_name = virtual_meta[oid]["name"]
                obj_path = os.path.join(data_path, "OakInkVirtualObjectsV2")
            obj_mesh_path = list(glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")))
            if len(obj_mesh_path) > 1:
                obj_mesh_path = [p for p in obj_mesh_path if "align" in os.path.split(p)[1]]
            my_obj_path = obj_mesh_path[0]



            obj_trimesh = trimesh.load(my_obj_path, process=False, force="mesh", skip_materials=True)
            bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
            obj_trimesh.vertices = obj_trimesh.vertices - bbox_center

            for i in range(outputs.shape[0]):
                outputs[i] = self.trans_denormalize(outputs[i].cpu()) # 归一化
                hand_pose = outputs[i, :48].cpu().numpy()
                hand_shape = outputs[i, 48:58].cpu().numpy()
                hand_transl_obj = outputs[i, 58:].cpu().numpy()
                mano_output = mano_layer(torch.from_numpy(hand_pose).unsqueeze(0).float(),torch.from_numpy(hand_shape).unsqueeze(0).float()).verts.squeeze(0).numpy()
                hand_verts_obj = mano_output+ hand_transl_obj
                mesh = trimesh.Trimesh(vertices=hand_verts_obj, faces=hand_faces)
                vis_data = [plot_mesh(obj_trimesh, color='lightblue')]
                vis_data += [plot_mesh(mesh, opacity=1.0, color='pink')]
                save_path = os.path.join(save_dir, 'html', f'{object_name}-{oid}+task_id-{int_name}+sample-{i}.html')
                layout = go.Layout(
                            scene=dict(
                                xaxis=dict(visible=False),  
                                yaxis=dict(visible=False),  
                                zaxis=dict(visible=False),  
                                bgcolor='rgba(0,0,0,0)'    
                            ),
                            margin=dict(l=0, r=0, t=0, b=0) 
                        )
                fig = go.Figure(data=vis_data, layout=layout)
                fig.write_html(save_path) 

                pbar.update(1)
            res['sample_qpos'][f"{object_name}-{int_name}-{oid}"] = np.array(outputs.cpu().detach())
        pickle.dump(res, open(os.path.join(save_dir, 'res_diffuser.pkl'), 'wb'))



@VISUALIZER.register()
@torch.no_grad()
class PoseGenVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for pose generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample

    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
    ) -> Any:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        
        Return:
            Results for gradio rendering.
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, D>
            
            i = 0
            scene_id = data['scene_id'][i]
            cam_tran = data['cam_tran'][i]
            gender = data['gender'][i]
            
            origin_cam_tran = data['origin_cam_tran'][i]
            scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
            scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
            scene_mesh.apply_transform(scene_trans)

            ## calculate camera pose
            camera_pose = np.eye(4)
            camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
            camera_pose = cam_tran @ camera_pose

            ## generate smplx bodies in last denoising step
            ## only visualize the body in last step, but visualize multi bodies
            smplx_params = outputs[i, :, -1, ...] # <k, ...>
            body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
            body_verts = body_verts.numpy()

            res_images = []
            for j in range(len(body_verts)):
                body_mesh = trimesh.Trimesh(vertices=body_verts[j], faces=body_faces)
                ## render generated body separately
                img = render_prox_scene({'scenes': [scene_mesh], 'bodies': [body_mesh]}, camera_pose, None)
                res_images.append(img)
            return res_images

@VISUALIZER.register()
@torch.no_grad()
class MotionGenVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for motion generation task.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
    
    def visualize(
        self, 
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        
        Return:
            Results for gradio rendering.
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, L, D>

            i = 0
            scene_id = data['scene_id'][i]
            cam_tran = data['cam_tran'][i]
            gender = data['gender'][i]

            origin_cam_tran = data['origin_cam_tran'][i]
            scene_trans = cam_tran @ np.linalg.inv(origin_cam_tran) # scene_T @ origin_cam_T = cur_cam_T
            scene_mesh = dataloader.dataset.scene_meshes[scene_id].copy()
            scene_mesh.apply_transform(scene_trans)

            ## calculate camera pose
            camera_pose = np.eye(4)
            camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
            camera_pose = cam_tran @ camera_pose

            ## generate smplx bodies in all denoising step
            ## only visualize the body in last step, visualize with gif
            smplx_params = outputs[i, :, -1, ...] # <k, ...>
            body_verts, body_faces, body_joints = dataloader.dataset.SMPLX.run(smplx_params, gender)
            body_verts = body_verts.numpy()
            
            res_ksamples = []
            for k in range(len(body_verts)):
                res_images = []
                for j, body in enumerate(body_verts[k]):
                    body_mesh = trimesh.Trimesh(vertices=body, faces=body_faces)
                    img = render_prox_scene({'scenes': [scene_mesh], 'bodies': [body_mesh]}, camera_pose, None)
                    res_images.append(img)
                res_ksamples.append(res_images)
            return res_ksamples


@VISUALIZER.register()
@torch.no_grad()
class PathPlanningRenderingVisualizerHF():
    def __init__(self, cfg: DictConfig) -> None:
        """ Visual evaluation class for path planning task. Directly rendering images.

        Args:
            cfg: visuzalizer configuration
        """
        self.ksample = cfg.ksample
        self.scannet_mesh_dir = cfg.scannet_mesh_dir
    
    def visualize(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """ Visualize method

        Args:
            model: diffusion model
            dataloader: test dataloader
        """
        model.eval()
        device = model.device

        for data in dataloader:
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)
            data['normalizer'] = dataloader.dataset.normalizer
            data['repr_type'] = dataloader.dataset.repr_type
            
            outputs = model.sample(data, k=self.ksample) # <B, k, T, L, D>

            scene_id = data['scene_id']
            trans_mat = data['trans_mat']
            target = data['target'].cpu().numpy()
            i = 0

            ## load scene and camera pose
            scene_mesh = trimesh.load(os.path.join(
                self.scannet_mesh_dir, 'mesh', f'{scene_id[i]}_vh_clean_2.ply'))
            scene_mesh.apply_transform(trans_mat[i])
            camera_pose = np.eye(4)
            camera_pose[0:3, -1] = np.array([0, 0, 10])

            sequences = outputs[i, :, -1, ...] # <k, horizon, 2>
            res_images = []
            for t in range(len(sequences)):
                path = sequences[t].cpu().numpy() # <horizon, 2>

                img = render_scannet_path(
                    {'scene': scene_mesh,
                    'target': create_trimesh_node(target[i], color=np.array([0, 255, 0], dtype=np.uint8)),
                    'path': create_trimesh_nodes_path(path, merge=True)},
                    camera_pose=camera_pose,
                    save_path=None
                )
                res_images.append(img)
            
            return res_images

def create_visualizer(cfg: DictConfig) -> nn.Module:
    """ Create a visualizer for visual evaluation
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A visualizer
    """
    return VISUALIZER.get(cfg.name)(cfg)

