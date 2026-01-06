import os
import sys
sys.path.append(os.getcwd())

import gc
import yaml
import pickle
import argparse
from loguru import logger


import torch
import random
import numpy as np
from typing import Dict
import trimesh as tm
from utils.handmodel import get_handmodel, compute_collision

from scripts.grasp_PPT.simulate import run_simulation
import scipy.cluster
from scipy.stats import entropy
import numpy as np
import torch
import json
from manotorch.manolayer import ManoLayer
import trimesh
import glob
from scipy.spatial.distance import cdist

def set_global_seed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test Scripts of Grasp Generation')
    parser.add_argument('--eval_dir', type=str, required=True, 
                        help='evaluation directory path (e.g.,\
                             "outputs/2022-11-15_18-07-50_GPUR_l1_pn2_T100/eval/final/2023-04-20_13-06-44")')
    
    return parser.parse_args()


def diversity_tester(args: argparse.Namespace, stability_results: dict) -> None:    
    grasps = pickle.load(open(os.path.join(args.eval_dir, 'res_diffuser.pkl'), 'rb'))

    qpos_std = []
    for object_name in grasps['sample_qpos'].keys():
        i_qpos = grasps['sample_qpos'][object_name][:, 9:]
        i_qpos = i_qpos[stability_results[object_name]['case_list'], :]
        if i_qpos.shape[0]:
            i_qpos = np.sqrt(i_qpos.var(axis=0))
            qpos_std.append(i_qpos)

    qpos_std = np.stack(qpos_std, axis=0)
    qpos_std = qpos_std.mean(axis=0).mean()
    logger.info(f'**Diversity** (std: rad.) across all success grasps: {qpos_std}')


def diversity(params_list, cls_num=20):
    # params_list = scipy.cluster.vq.whiten(params_list)
    #  # k-means
    codes, dist = scipy.cluster.vq.kmeans(params_list, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(params_list, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences  count: [20]
    print(counts)
    ee = entropy(counts)
    return ee, np.mean(dist)

    kps_all = np.stack(kps_all_list, axis=0)  # [120, 63]
    cls_num = 20

def diversity_tester_wo_success(eval_dir) -> None:    
    grasps = pickle.load(open(os.path.join(eval_dir, 'res_diffuser.pkl'), 'rb'))

    pose_std = []
    shape_std = []
    transl_obj_std = []
    rot_std = []
    joint_std = []


    pose_list = []
    for object_name in grasps['sample_qpos'].keys():
        i_pose = grasps['sample_qpos'][object_name][:, :48]
        i_shape = grasps['sample_qpos'][object_name][:, 48:58]
        i_transl_obj = grasps['sample_qpos'][object_name][:, 58:]

        i_rot = grasps['sample_qpos'][object_name][:, :3]
        i_joint = grasps['sample_qpos'][object_name][:, 3:48]


        all = grasps['sample_qpos'][object_name][:, :]
        pose_list.extend(all)
    

        if i_pose.shape[0]:
            i_pose = np.sqrt(i_pose.var(axis=0))
            i_rot = np.sqrt(i_rot.var(axis=0))
            i_joint = np.sqrt(i_joint.var(axis=0))
            i_shape = np.sqrt(i_shape.var(axis=0))
            i_transl_obj = np.sqrt(i_transl_obj.var(axis=0))
            
            pose_std.append(i_pose)
            rot_std.append(i_rot)
            joint_std.append(i_joint)

            shape_std.append(i_shape)
            transl_obj_std.append(i_transl_obj)

    pose_std = np.stack(pose_std, axis=0)
    shape_std = np.stack(shape_std, axis=0)
    transl_obj_std = np.stack(transl_obj_std, axis=0)

    pose_matrix = np.array(pose_list)
    print("pose_matrix shape is ", pose_matrix.shape)
    n_clusters = 20
    grasp_entropy, grasp_dist = diversity(pose_matrix, cls_num=n_clusters)
    print("抓取多样性信息熵 (entropy):", grasp_entropy)
    print("抓取聚类平均距离 (dist):", grasp_dist)

    pose_std = pose_std.mean(axis=0).mean()
    shape_std = shape_std.mean(axis=0).mean()
    transl_obj_std = transl_obj_std.mean(axis=0).mean()

    rot_std = np.stack(rot_std, axis=0)
    rot_std = rot_std.mean(axis=0).mean()

    joint_std = np.stack(joint_std, axis=0)
    joint_std = joint_std.mean(axis=0).mean()


    logger.info(f'**Diversity** (std: rad.) across all grasps rot_std: {rot_std}, joint_std: {joint_std}')


def collision_tester(args: argparse.Namespace, stability_results: dict) -> None:
    _BATCHSIZE = 16 #NOTE: adjust this batchsize to fit your GPU memory && need to be divided by generated grasps per object
    _NPOINTS = 4096 #NOTE: number of surface points sampled from a object

    grasps = pickle.load(open(os.path.join(args.eval_dir, 'res_diffuser.pkl'), 'rb'))
    obj_pcds_nors_dict = pickle.load(open('/home/puhao/data/MultiDex_UR/object_pcds_nors.pkl', 'rb'))
    hand_model = get_handmodel(batch_size=_BATCHSIZE, device=args.device)

    collisions_dict = {obj: [] for obj in grasps['sample_qpos'].keys()}
    for object_name in grasps['sample_qpos'].keys():
        qpos = grasps['sample_qpos'][object_name]
        obj_pcd_nor = obj_pcds_nors_dict[object_name][:_NPOINTS, :]
        
        for i in range(qpos.shape[0] // _BATCHSIZE):
            i_qpos = qpos[i * _BATCHSIZE: (i + 1) * _BATCHSIZE, :]
            hand_model.update_kinematics(q=torch.tensor(i_qpos, device=args.device))
            hand_surface_points = hand_model.get_surface_points()
            #TODO: needed to be checked
            depth_collision = compute_collision(torch.tensor(obj_pcd_nor, device=args.device), hand_surface_points)
            collisions_dict[object_name].append(np.array(depth_collision.cpu()[stability_results[object_name]['case_list'][i * _BATCHSIZE : (i + 1) * _BATCHSIZE]]))
        collisions_dict[object_name] = np.concatenate(collisions_dict[object_name], axis=0)
    
    collision_values = np.concatenate([collisions_dict[object_name] for object_name in grasps['sample_qpos'].keys()], axis=0)
    logger.info(f'**Collision** (depth: mm.) across all grasps: {collision_values.mean() * 1e3}')

def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="") -> None:
        super().__init__()
        self.reset()
        self.name = name

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_by_mean(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.avg:.4e}"

    def get_measures(self) -> Dict:
        return {f"{self.name} avg": self.avg}



def penetration(obj_verts, obj_faces, hand_verts, mode="max"):
    from thirdparty.libmesh.inside_mesh import check_mesh_contains
    obj_trimesh = trimesh.Trimesh(vertices=np.asarray(obj_verts), faces=np.asarray(obj_faces))
    inside = check_mesh_contains(obj_trimesh, hand_verts)

    valid_vals = inside.sum()
    if valid_vals > 0:
        selected_hand_verts = hand_verts[inside, :]

        mins_sel_hand_to_obj = np.min(cdist(selected_hand_verts, obj_verts), axis=1)

        collision_vals = mins_sel_hand_to_obj
        if mode == "max":
            penetr_val = np.max(collision_vals)  # max
        elif mode == "mean":
            penetr_val = np.mean(collision_vals)
        elif mode == "sum":
            penetr_val = np.sum(collision_vals)
        else:
            raise KeyError("unexpected penetration mode")
    else:
        penetr_val = 0
    return penetr_val


def eval(eval_dir):

    grasps = pickle.load(open(os.path.join(eval_dir, 'res_diffuser.pkl'), 'rb'))

    mano_layer = ManoLayer(
            rot_mode="axisang",
            center_idx=9,
            mano_assets_root="assets/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        )
    rh_faces = mano_layer.get_mano_closed_faces().numpy()


    pentr_dep_sum = AverageMeter("pentr_dep")
    pentr_vol_sum = AverageMeter("pentr_vol")
    sims_disp_sum = AverageMeter("sims_disp")
    sims_disp_var = []
    diversity = diversity_tester_wo_success(eval_dir)
    base_path = "/home/syks/Scene-Diffuser-obj/data/OakInk/shape"

    meta_path = os.path.join(base_path, "metaV2")
    real_meta = json.load(open(os.path.join(meta_path, "object_id.json"), "r"))
    virtual_meta = json.load(open(os.path.join(meta_path, "virtual_object_id.json"), "r"))
    num_coll = 0
    num_sum = 0
    for output in grasps['sample_qpos'].keys():
        obj_name, intent_name, oid = output.split("-")
        cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.]]).astype(np.float32)
        if oid in real_meta:
            object_name = real_meta[oid]["name"]
            obj_path = os.path.join(base_path, "OakInkObjectsV2")
        else:
            object_name = virtual_meta[oid]["name"]
            obj_path = os.path.join(base_path, "OakInkVirtualObjectsV2")
        obj_mesh_path = list(glob.glob(os.path.join(obj_path, object_name, "align_ds", "*.obj")))

        if len(obj_mesh_path) > 1:
            obj_mesh_path = [p for p in obj_mesh_path if "align" in os.path.split(p)[1]]
        
        obj_trimesh = trimesh.load(obj_mesh_path[0], process=False, force="mesh", skip_materials=True)
        bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
        obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
        obj_mesh_verts = obj_trimesh.vertices
        origin_faces = obj_trimesh.faces
        obj_mesh_verts = obj_mesh_verts.dot(cam_extr[:3,:3].T)  # [N,3]
        obj_mesh = trimesh.Trimesh(vertices=obj_mesh_verts,
                                    faces=origin_faces)  # obj
        for i in range(len(grasps['sample_qpos'][output])):
            hand_pose = grasps['sample_qpos'][output][i, :48]
            hand_shape = grasps['sample_qpos'][output][i, 48:58]
            hand_transl_obj = grasps['sample_qpos'][output][i, 58:]

            mano_output = mano_layer(torch.from_numpy(hand_pose).unsqueeze(0).float(),torch.from_numpy(hand_shape).unsqueeze(0).float()).verts.squeeze(0).numpy()
            hand_verts_obj = mano_output+ hand_transl_obj
            final_mano_verts = hand_verts_obj.dot(cam_extr[:3,:3].T)

            hand_mesh = trimesh.Trimesh(vertices=final_mano_verts, faces=rh_faces)
            num_sum =  num_sum+1
            pentr_dep = penetration(obj_verts=obj_mesh_verts, obj_faces=origin_faces, hand_verts=final_mano_verts)
            if pentr_dep > 0.005:
                num_coll = num_coll+1
            pentr_dep_sum.update(pentr_dep)
            penetr_vol = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
            pentr_vol_sum.update(penetr_vol)
            vhacd_exe = "/home/syks/Scene-Diffuser-obj/thirdparty/v-hacd-master/app/build/TestVHACD"
            simu_disp = run_simulation(final_mano_verts, rh_faces.reshape((-1, 3)),
                                            obj_mesh_verts, origin_faces,
                                            vhacd_exe=vhacd_exe)
            sims_disp_sum.update(simu_disp)
            sims_disp_var.append(simu_disp)


       
    print("pentr_vol_sum get_measures is ", pentr_vol_sum.avg)
    print("pentr_dep_sum get_measures is ", pentr_dep_sum.avg)
    print("sims_disp_sum get_measures is ", sims_disp_sum.avg)
    print("sims_disp_var is ", np.std(sims_disp_var))

    output_file = os.path.join(eval_dir, "new_eval.txt")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"pentr_vol_sum mean: {pentr_vol_sum.avg}\n")
        file.write(f"pentr_dep_sum mean: {pentr_dep_sum.avg}\n")
        file.write(f"sims_disp_sum mean: {sims_disp_sum.avg}\n")
        file.write(f"sims_disp_var std: {np.std(sims_disp_var)}\n")

    print(f"Evaluation done!, results are saved in: {output_file}")
    print(f"num_coll is  { num_coll},  num_sum is {num_sum}")
    return 




def main() -> None:
    args = parse_args()
    eval(args.eval_dir)
    
    logger.info('End evaluating..')


if __name__ == '__main__':
    main()
