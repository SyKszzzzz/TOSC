from typing import Dict
import os
import numpy as np
import torch
from omegaconf import DictConfig
from utils.handmodel import get_handmodel
from models.optimizer.optimizer import Optimizer
from models.base import OPTIMIZER
import pickle
import torch.functional as F
import json
import glob
import trimesh
from manotorch.manolayer_cuda import ManoLayer
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes


def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign


@OPTIMIZER.register()
class Grasp_ADT(Optimizer):

    _BATCH_SIZE = 10
    _N_OBJ = 4096
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
         1.7509e-01], device='cuda')


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
        -1.4180e-01], device='cuda')

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self.slurm = slurm
        self.scale = cfg.scale
        self.collision = cfg.collision
        self.collision_weight = cfg.collision_weight
        self.clip_grad_by_value = cfg.clip_grad_by_value

        self.modeling_keys = cfg.modeling_keys

        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans

        self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        # self.obj_pcds_nors_dict = pickle.load(open(os.path.join(self.asset_dir, 'object_pcds_nors.pkl'), 'rb'))
        # self.hand_model = get_handmodel(batch_size=self._BATCH_SIZE, device=self.device)

        self.relu = torch.nn.ReLU()

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            center_idx=9,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )

    def optimize(self, x: torch.Tensor, data: Dict, t: int) -> torch.Tensor:
        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step, which is detached and is required grad
            data: data dict that provides original data
            t: sample time

        Return:
            The optimizer objective value of current step
        """
        loss = 0.
        # print(x)

        x = self.trans_denormalize(x)
        hand_pose = x[:, :48]
        hand_shape = x[:, 48:58]
        hand_transl_obj = x[:, 58:]
       
        final_mano_verts_list = []
        for i in range(x.shape[0]):
            print("hand_pose[i, :].unsqueeze(0).float() shape is ", hand_pose[i, :].unsqueeze(0).float().shape)
            print("hand_shape[i, :].unsqueeze(0).float() shape is ", hand_shape[i, :].unsqueeze(0).float().shape)
            mano_output = self.mano_layer(hand_pose[i, :].unsqueeze(0).float(), hand_shape[i, :].unsqueeze(0).float()).verts.squeeze(0)
            hand_verts_obj = mano_output+ hand_transl_obj[i,:]
            final_mano_verts_list.append(hand_verts_obj)

        final_mano_verts = torch.stack(final_mano_verts_list, axis=0)
        # final_mano_verts = torch.tensor(final_mano_verts, device='cuda')
        n_hand = final_mano_verts.shape[1]



        print("final_mano_verts shape is ", final_mano_verts.shape)
        oid = data['scene_id'][0] # 物体点云
        obj_suffix_path = "align"
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
        mesh = Meshes(verts=torch.from_numpy(obj_trimesh.vertices).unsqueeze(0).float(), faces=torch.from_numpy(obj_trimesh.faces).unsqueeze(0).float())
        obj_verts_ds, obj_normals_ds = sample_points_from_meshes(mesh, self._N_OBJ, return_normals=True)
        # obj_pcd_list = []
        # obj_nor_list = []
        print("obj_verts_ds shape is ", obj_verts_ds.shape , "obj_normals_ds shape is ", obj_normals_ds)
        # for i in range(x.shape[0]):
        #     obj_pcd_list.append(obj_verts_ds)
        #     obj_nor_list.append(obj_normals_ds)
        obj_pcd = obj_verts_ds.repeat(self._BATCH_SIZE, 1, 1).to("cuda")

        obj_nor = obj_normals_ds.repeat(self._BATCH_SIZE, 1, 1).to("cuda")
        # obj_pcd = torch.tensor(obj_pcd, device='cuda')

        # obj_nor = np.stack(obj_nor_list, axis=0)
        # obj_nor = torch.tensor(obj_nor, device='cuda')
        # print(" final_mano_verts shape is ", final_mano_verts.shape)
        print("n_hand is ", n_hand, "obj_nor shape is ", obj_nor.shape, "obj_pcd shape is ", obj_pcd.shape)

        batch_obj_pcd = obj_pcd[:, :, :3].view(self._BATCH_SIZE, 1, self._N_OBJ, 3).repeat(1, n_hand, 1, 1)
        batch_hand_pcd = final_mano_verts.view(self._BATCH_SIZE, n_hand, 1, 3).repeat(1, 1, self._N_OBJ, 1)
        hand_obj_dist = (batch_obj_pcd - batch_hand_pcd).norm(dim=3)
        hand_obj_dist, hand_obj_indices = hand_obj_dist.min(dim=2)
        hand_obj_points = torch.stack([obj_pcd[i, x, :] for i, x in enumerate(hand_obj_indices)], dim=0)
        hand_obj_normals = torch.stack([obj_nor[i, x, :] for i, x in enumerate(hand_obj_indices)], dim=0)
        # compute the signs
        hand_obj_signs = ((hand_obj_points - final_mano_verts) * hand_obj_normals).sum(dim=2)
        hand_obj_signs = (hand_obj_signs > 0.).float()
        # signs dot dist to compute collision value
        # collision_value = (hand_obj_signs * hand_obj_dist).max(dim=1).values
        collision_value = (hand_obj_signs * hand_obj_dist).sum(dim=1)
        # collision_value = self.relu(collision_value - 0.1)
        # collision_value = torch.abs(collision_value - 0.005)
        loss += self.collision_weight * collision_value.mean()

        # signed_distance_list = []
        # for i in range(x.shape[0]):
        #     signed_distance = obj_trimesh.nearest.signed_distance(final_mano_verts[i].detach().cpu().numpy())
        #     signed_distance_list.append(signed_distance)

        # signed_distance_list = torch.tensor(np.stack(signed_distance_list, axis=0), device='cuda').requires_grad_(True)
        # print("signed_distance_list shape is ", signed_distance_list.shape)
        # collision_value = signed_distance_list.sum(dim = 1)
        # print("collision_value shape is ", collision_value.shape)


        # loss += self.collision_weight * collision_value.mean()
        # print("loss is ", loss)

        return (-1.0) * loss
    
        

    def gradient(self, x: torch.Tensor, data: Dict, variance: torch.Tensor) -> torch.Tensor:
        # print(f'compute gradient...')
        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step
            data: data dict that provides original data

        Return:
            Commputed gradient
        """
        assert (x.shape[0] % self._BATCH_SIZE == 0)
        print("x shape is ", x.shape)
        # print("data key is ", data)
        with torch.enable_grad():
            # concatenate the id rot to x_in
            
            x_in = x.detach().requires_grad_(False)
            grad_list = []
            obj_list = []
            for i in range(x.shape[0] // self._BATCH_SIZE):
                i_x_in = x_in[i*self._BATCH_SIZE:(i+1)*self._BATCH_SIZE, :].detach().requires_grad_(True)
                
                obj = self.optimize(i_x_in, data, t=i)
                i_grad = torch.autograd.grad(obj, i_x_in, allow_unused=True)[0]
                print("i_grad is ", i_grad)
                
                obj_list.append(obj.abs().mean().detach().cpu())
                grad_list.append(i_grad)
            # print(f'loss: {np.mean(obj_list)}')
            grad = torch.cat(grad_list, dim=0)
            ## clip gradient by value
            # print(f'grad norm: {grad.abs().mean()}')
            grad = grad * self.scale
            grad = torch.clip(grad, **self.clip_grad_by_value)
            # grad = torch.cat([grad[:, :3], grad[:, 9:]], dim=-1)
            # grad = torch.cat([torch.zeros_like(grad[:, :3], device=self.device), grad[:, 9:]], dim=-1)
            # grad = torch.cat([torch.zeros_like(grad[:, :3], device=self.device),
            #                   torch.zeros_like(grad[:, 9:11], device=self.device),
            #                   grad[:, 11:]], dim=-1)
            return grad



    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self.x_upper - self.x_lower) + self.x_lower
        return global_trans_denorm