from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import numpy as np
from models.base import DIFFUSER
from models.dm.schedule import make_schedule_ddpm
from models.optimizer.optimizer import Optimizer
from models.planner.planner import Planner
import trimesh
from manotorch.manolayer import ManoLayer
# from models.optimizer.grasp_loss_pose import GraspLossPose
from csdf import compute_sdf, index_vertices_by_faces
from pytorch3d.loss import chamfer_distance
import math
import warnings
from functools import partial
from typing import Optional, Union
from pytorch3d.structures import Meshes
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.loss import point_mesh_face_distance
# from pytorch3d.ops import point_mesh_face_distance
import numpy as np
import torch
# from pytorch3d.ops import ray_mesh_intersect
from pytorch3d.ops.knn import knn_gather, knn_points

def optimized_scale(positive_flat, negative_flat):
  # Calculate dot production
  dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

  # Squared norm of uncondition
  squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

  # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
  st_star = dot_product / squared_norm
  return st_star

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

def inter_penetr_loss(hand_xyz, hand_face, obj_xyz, nn_dist, nn_idx):
    '''
    计算手部和物体之间穿透损失的函数。
    get penetrate object xyz and the distance to its NN
    :param hand_xyz: [B, 778, 3]
    :param hand_face: [B, 1538, 3], hand faces vertex index in [0:778]
    :param obj_xyz: [B, 3000, 3]
    :param nn_idx: [B, 3000], index of NN in hand vertices from obj vertices
    :return: inter penetration loss
    '''
    B = hand_xyz.size(0)
    mesh = Meshes(verts=hand_xyz, faces=hand_face)
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

    # if not nn_dist:
    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
    
    #检测内部点 通过检测物体点云位于手部mesh内部的点来判断穿透
    interior = get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)  # True for interior
    # 计算穿透损失
    penetr_dist = nn_dist[interior].sum() / B  # batch reduction
    return penetr_dist #! 乘以100.0是为了放大损失，使得穿透损失的影响更大  目标是最小化手部和物体之间的穿透程度


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

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


def trans_denormalize(global_trans: torch.Tensor):
    global_trans_denorm = global_trans + (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
    global_trans_denorm /= (_NORMALIZE_UPPER - _NORMALIZE_LOWER)
    global_trans_denorm = global_trans_denorm * (x_upper - x_lower) + x_lower
    return global_trans_denorm

_NORMALIZE_LOWER = -1.
_NORMALIZE_UPPER = 1.
import time


@DIFFUSER.register()
class FlowGrasp(nn.Module):
    def __init__(self, eps_model: nn.Module, cfg: DictConfig, has_obser: bool, *args, **kwargs) -> None:
        super(FlowGrasp, self).__init__()
        
        self.eps_model = eps_model # eps model, UNetModel
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type
        self.sigma = 0.0
        self.has_observation = has_obser # used in some task giving observation

        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)
        self.other = False
        if cfg.loss_type == 'l1':
            self.criterion = F.l1_loss
        elif cfg.loss_type == 'l2':
            self.criterion = F.mse_loss
        else:
            raise Exception('Unsupported loss type.')
                
        self.optimizer = None
        self.planner = None
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            center_idx=9,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )
    @property
    def device(self):
        return self.betas.device
    
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0
    
    def sample_noise_like(self, x):
        return torch.randn_like(x)
    
    def compute_sigma_t(self, t):
        del t
        return self.sigma
    
    def sample_xt(self, x0, x1, t, epsilon):
        mu_t = self.compute_mu_t(x0, x1, t)  # sigma是0,所以实际上的xt就是t * x1 + (1 - t) * x0，就是这个的输出
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)
    

    def forward(self, data: Dict) -> torch.Tensor:
        data['x'] = torch.tensor(data['x'], device=self.device)
        B = data['x'].shape[0]

        x1 = data['x']
        x0 = torch.randn_like(x1)
        

        start_time = time.time()
        condtion = self.eps_model.condition(data)   
        print("--- %s seconds ---" % (time.time() - start_time))
        t, xt, ut, eps = self.sample_location_and_conditional_flow(x0, x1, return_noise=True)
        xt.requires_grad_(True)
        vt = self.eps_model(xt.detach(), t, condtion)
        para_loss = torch.mean((vt - ut) ** 2)
        hand_faces = self.mano_layer.get_mano_closed_faces()
        time_factor = (1 - t).unsqueeze(1)
        pred_xt = vt*time_factor + xt

        for i in range(pred_xt.shape[0]):
            pred_xt[i] = trans_denormalize(pred_xt[i].cpu()) # 归一化
        hand_pose = pred_xt[..., :48]
        hand_shape = pred_xt[..., 48:58]
        hand_transl_obj = pred_xt[..., 58:]
        mano_output = self.mano_layer(hand_pose.float(), hand_shape.float())

        targets_hand_pose = torch.tensor(data['hand_pose_obj']).to('cuda')
        targets_hand_shape = torch.tensor(data['hand_shape']).to('cuda')
        targets_mano_output = self.mano_layer(targets_hand_pose.float(), targets_hand_shape.float())

        pred_hand_pc = mano_output.verts
        target_hand_pc = targets_mano_output.verts
        chamfer_loss = chamfer_distance(pred_hand_pc, target_hand_pc, point_reduction="sum", batch_reduction="mean")[0]

        batch_size = data["x"].size(0)
        point_object = torch.tensor(data['pos_ori']).float().to('cuda') 
        hand_verts_obj = (pred_hand_pc + hand_transl_obj.unsqueeze(1)) 
        obj_nn_dist_recon, obj_nn_idx_recon = get_NN(point_object, hand_verts_obj)
        hand_faces_batch = hand_faces.unsqueeze(0).expand(hand_verts_obj.shape[0], -1, -1)
        penetr_loss = inter_penetr_loss(hand_verts_obj, hand_faces_batch.to('cuda'), point_object,
                                        obj_nn_dist_recon, obj_nn_idx_recon)
        pred_hand_keypoints = mano_output.joints
        dis_spen = (pred_hand_keypoints.unsqueeze(1) - pred_hand_keypoints.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
        dis_spen = 0.02 - dis_spen
        dis_spen[dis_spen < 0] = 0
        loss_spen = dis_spen.sum() / batch_size

        loss_phys =  1.*chamfer_loss + 6*loss_spen + 6.*penetr_loss
        alpha_t = (1 - t).unsqueeze(1)
        g_t, = torch.autograd.grad(
          loss_phys, xt,
          create_graph=False,   
          retain_graph=True)   
        
        v_tar = ut - alpha_t * g_t.detach()
        loss = F.mse_loss(vt, v_tar)
        return {'loss': loss, 'chamfer_loss': chamfer_loss, 'para_loss':para_loss, 'loss_pen': penetr_loss, 'loss_spen': loss_spen}
    
    @torch.no_grad()
    def feature(self, data: Dict, k: int=1) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        
        condtion = self.eps_model.condition(data)
        x_rec   = condtion[:, :128, :]
        return x_rec

    @torch.no_grad()
    def sample(self, data: Dict, k: int=1) -> torch.Tensor:
        def optimized_scale(positive_flat, negative_flat):
            # Calculate dot production
            dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

            # Squared norm of uncondition
            squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

            # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
            st_star = dot_product / squared_norm
            return st_star
        
        ksamples = []
        for index in range(k):
            timehorion = 10
            x_t = data['x']
            condtion = self.eps_model.condition(data)
            for i in range(timehorion):
                timestep = torch.tensor([i / timehorion]).to(self.device)
                if i == 0:
                    vt = self.eps_model(x_t, timestep, condtion)
                    traj = (vt * 1 / timehorion + x_t)

                else:
                    vt = self.eps_model(traj, timestep, condtion)
                    traj = (vt * 1 / timehorion + traj)
            ksamples.append(traj)

        ksamples = torch.stack(ksamples, dim=1)
        return ksamples
    
    def set_optimizer(self, optimizer: Optimizer):
        """ Set optimizer for diffuser, the optimizer is used in sampling

        Args:
            optimizer: a Optimizer object that has a gradient method
        """
        self.optimizer = optimizer
    
    def set_planner(self, planner: Planner):
        """ Set planner for diffuser, the planner is used in sampling

        Args:
            planner: a Planner object that has a gradient method
        """
        self.planner = planner

    

    