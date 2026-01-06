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
from manotorch.manolayer import ManoLayer
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import chamfer_distance
# from emd import earth_mover_distance

from pytorch3d.structures import Meshes


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

def get_faces_xyz(faces_idx, xyz):
    '''
    :param faces_idx: [B, N1, 3]. N1 is number of faces (1538 for MANO), index of face vertices in N2
    :param xyz: [B, N2, 3]. N2 is number of points.
    :return: faces_xyz: [B, N1, 3, 3] faces vertices coordinate
    '''
    B, N1, D = faces_idx.size()
    N2 = xyz.size(1)
    xyz_replicated = xyz.cpu().unsqueeze(1).repeat(1,N1,1,1)  # use cpu to save CUDA memory
    faces_idx_replicated = faces_idx.unsqueeze(-1).repeat(1,1,1,D).type(torch.LongTensor)
    return torch.gather(xyz_replicated, dim=2, index=faces_idx_replicated).to(faces_idx.device)

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

def CVAE_loss_mano(recon_x, x, mean, log_var, loss_tpye, mode='train'):
    '''
    :param recon_x: reconstructed hand xyz [B,778,3]
    :param x: ground truth hand xyz [B,778,6]
    :param mean: [B,z]
    :param log_var: [B,z]
    :return:
    '''
    
    recon_loss, _ = chamfer_distance(recon_x, x, point_reduction='sum', batch_reduction='mean')
    # elif loss_tpye == 'EMD':
    #     recon_loss = earth_mover_distance(recon_x, x, transpose=False).sum() / x.size(0)
    # if mode != 'train':
    #     return recon_loss
    # # KLD loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0) * 10.0
    # if mode == 'train':
    return recon_loss + KLD, recon_loss.item(), KLD.item()
    

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

def trans_denormalize(global_trans: torch.Tensor):
    global_trans_denorm = global_trans + (_NORMALIZE_UPPER - _NORMALIZE_LOWER) / 2
    global_trans_denorm /= (_NORMALIZE_UPPER - _NORMALIZE_LOWER)
    global_trans_denorm = global_trans_denorm * (x_upper - x_lower) + x_lower
    return global_trans_denorm

_NORMALIZE_LOWER = -1.
_NORMALIZE_UPPER = 1.

def CMap_loss3(obj_xyz, hand_xyz, cmap):
    '''
    # prior cmap loss on gt cmap
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N1] for contact map from NN dist thresholding
    :param hand_faces_index: [B, 1538, 3] hand index in [0,N2] for 3 vertices in a face
    :return:
    '''

    # finger_vertices = [309, 317, 318, 319, 320, 322, 323, 324, 325,
    #                    326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
    #                    350, 351, 352, 353, 354, 355,  # 2nd finger
    #                    429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 461, 462, 463, 465, 466,
    #                    467,  # 3rd
    #                    547, 548, 549, 550, 553, 566, 573, 578,  # 4th
    #                    657, 661, 662, 664, 665, 666, 667, 670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691,
    #                    692, 693, 694, 695,  # 5th
    #                    736, 737, 738, 739, 740, 741, 743, 753, 754, 755, 756, 757, 759, 760, 761, 762, 763, 764, 766,
    #                    767, 768,  # 1st
    #                    73, 96, 98, 99, 772, 774, 775, 777]  # hand
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

    # compute contact map loss
    n_points = torch.sum(cmap)
    cmap_loss = obj_CD[cmap].sum() / (B * n_points)

    return 3000.0 * cmap_loss

def CMap_consistency_loss(obj_xyz, recon_hand_xyz, gt_hand_xyz, recon_dists, gt_dists):
    '''
    :param recon_hand_xyz: [B, N2, 3]
    :param gt_hand_xyz: [B, N2, 3]
    :param obj_xyz: [B, N1, 3]
    :return:
    '''
    # if not recon_dists or not gt_dists:
    #     recon_dists, _ = utils_loss.get_NN(obj_xyz, recon_hand_xyz)  # [B, N1]
    #     gt_dists, _ = utils_loss.get_NN(obj_xyz, gt_hand_xyz)  # [B, N1]
    recon_dists = torch.sqrt(recon_dists)
    gt_dists = torch.sqrt(gt_dists)
    # hard cmap
    recon_cmap = recon_dists < 0.005
    gt_cmap = gt_dists < 0.005
    gt_cpoint_num = gt_cmap.sum() + 0.0001
    consistency = (recon_cmap * gt_cmap).sum() / gt_cpoint_num
    # soft cmap
    #consistency2 = torch.nn.functional.mse_loss(recon_dists, gt_dists, reduction='none').sum() / recon_dists.size(0)
    return -5.0 * consistency #+ consistency2

def inter_penetr_loss(hand_xyz, hand_face, obj_xyz, nn_dist, nn_idx):
    '''
    get penetrate object xyz and the distance to its NN
    :param hand_xyz: [B, 778, 3]
    :param hand_face: [B, 1538, 3], hand faces vertex index in [0:778]
    :param obj_xyz: [B, 3000, 3]
    :return: inter penetration loss
    '''
    B = hand_xyz.size(0)
    mesh = Meshes(verts=hand_xyz, faces=hand_face)
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

    # if not nn_dist:
    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
    interior = get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)  # True for interior
    penetr_dist = nn_dist[interior].sum() / B  # batch reduction
    return 100.0 * penetr_dist

@DIFFUSER.register()
class TTA(nn.Module):

    def __init__(self, eps_model: nn.Module, cfg: DictConfig, in_dim=61, cond_dim=1152, n_neurons=512, latentD=16, *args, **kwargs):
        super(TTA, self).__init__()
        # self.timesteps = cfg.steps
        # self.schedule_cfg = cfg.schedule_cfg
        # self.rand_t_type = cfg.rand_t_type

        self.obj_inchannel = 4 # 4
        self.cvae_encoder_sizes = [1024, 512, 256] # [1024, 512, 256]
        self.cvae_latent_size = 64 # 64
        self.cvae_decoder_sizes = [1024, 256, 61] # [1024, 256, 61]
        self.cvae_condition_size = 1024 # 1024
        self.eps_model = eps_model
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        self.optimizer = None
        self.planner = None
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            center_idx=9,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )
        self.hand_faces = self.mano_layer.get_mano_closed_faces().numpy()
    
    def forward(self, data):
        x = data['x'].to(self.device)
        B = data['x'].shape[0]
        rh_faces = torch.from_numpy(self.hand_faces).view(1,-1,3).contiguous()
        rh_faces = rh_faces.repeat(B, 1, 1).to(self.device)
        epoch = data['epoch']

        a = 1.0
        b = 0.1
        c = 1000.0
        d = 10.0
        e = 10.0
        for i in range(x.shape[0]):
            x[i] = trans_denormalize(x[i].cpu())

        x_hand_pose = x[..., :48]
        x_hand_shape = x[..., 48:58]
        x_hand_transl_obj = x[..., 58:]
        x_mano_output = self.mano_layer(x_hand_pose.float(), x_hand_shape.float())
        x_pred_hand_pc = x_mano_output.verts
        hand_xyz = (x_pred_hand_pc + x_hand_transl_obj.unsqueeze(1)) # 生成抓取的点云

        obj_pc = data['pos'].permute(0,2,1).float()

        obj_glb_feature, hand_glb_feature = self.eps_model.condition(hand_xyz.permute(0,2,1), obj_pc)

        recon_param, mean, log_var, z = self.eps_model(hand_glb_feature, obj_glb_feature)

        for i in range(recon_param.shape[0]):
            recon_param[i] = trans_denormalize(recon_param[i].cpu())

        out_hand_pose = recon_param[..., :48]
        out_hand_shape = recon_param[..., 48:58]
        out_hand_transl_obj = recon_param[..., 58:]
        out_mano_output = self.mano_layer(out_hand_pose.float(), out_hand_shape.float())
        out_pred_hand_pc = out_mano_output.verts
        recon_xyz = (out_pred_hand_pc + out_hand_transl_obj.unsqueeze(1))

        obj_nn_dist_gt, obj_nn_idx_gt = get_NN(obj_pc.permute(0,2,1)[:,:,:3], hand_xyz)
        obj_nn_dist_recon, obj_nn_idx_recon = get_NN(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)

        # mano param loss
        param_loss = torch.nn.functional.mse_loss(recon_param, x, reduction='none').sum() / recon_param.size(0)
        # mano recon xyz loss, KLD loss
        cvae_loss, recon_loss_num, KLD_loss_num = CVAE_loss_mano(recon_xyz, hand_xyz, mean, log_var, 'CD', 'train')
        # cmap loss
        #cmap_loss = CMap_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_cmap)
        cmap_loss = CMap_loss3(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_nn_dist_recon < 0.01**2)
        # cmap consistency loss
        consistency_loss = CMap_consistency_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, hand_xyz,
                                                 obj_nn_dist_recon, obj_nn_dist_gt)
        # inter penetration loss
        penetr_loss = inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0,2,1)[:,:,:3],
                                        obj_nn_dist_recon, obj_nn_idx_recon)
        if epoch >= 5:
            loss = a * cvae_loss + b * param_loss + c * cmap_loss + d * penetr_loss + e * consistency_loss
        else:
            loss = a * cvae_loss + b * param_loss + d * penetr_loss + e * consistency_loss

        return {'loss': loss, 'cvae_loss': cvae_loss, 'param_loss': param_loss, 'penetr_loss': penetr_loss, 'consistency_loss':consistency_loss}
    
    @torch.no_grad()
    def sample(self, data, seed=None, k: int=1):
        obj_pc = data['pos'].permute(0,2,1).float()
        B = obj_pc.size(0)
        
        recon = self.eps_model.inference(obj_pc)

        print(f"ksamples shape is {recon.shape}")
        return recon  
        