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
from typing import List, Optional, Tuple, Union
from scripts.grasp_gen_ur_oakink.simulate import run_simulation
import scipy.cluster
from scipy.stats import entropy
import numpy as np
import torch
import json
from manotorch.manolayer import ManoLayer
import trimesh
import glob
from scipy.spatial.distance import cdist
from tqdm import tqdm
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


def diversity_tester_wo_success(eval_dir) -> None:    
    grasps = pickle.load(open(os.path.join(eval_dir, 'res_diffuser.pkl'), 'rb'))

    pose_std = []
    shape_std = []
    transl_obj_std = []
    rot_std = []
    joint_std = []
    all_list = []


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

    
    pose_list = np.stack(pose_list, axis=0)
    pose_list = pose_list.mean(axis=0).mean()


    logger.info(f'**Diversity** (std: rad.) across all grasps rot_std: {rot_std}, joint_std: {joint_std}, pose: {pose_list}')






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
ALL_INTENT = {
"use": "0001",
"hold": "0002",
"liftup": "0003",
"handover": "0004",
}

"""
Evaluate P-FID between two batches of point clouds.

The point cloud batches should be saved to two npz files, where there
is an arr_0 key of shape [N x K x 3], where K is the dimensionality of
each point cloud and N is the number of clouds.
"""

import argparse

from point_e.evals.feature_extractor import PointNetClassifier, get_torch_devices
from point_e.evals.fid_is import compute_statistics
from point_e.evals.npz_stream import NpzStreamer


def p_fid():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("batch_1", type=str)
    parser.add_argument("batch_2", type=str)
    args = parser.parse_args()

    print("creating classifier...")
    clf = PointNetClassifier(devices=get_torch_devices(), cache_dir=args.cache_dir)

    print("computing first batch activations")

    features_1, _ = clf.features_and_preds(NpzStreamer(args.batch_1))
    stats_1 = compute_statistics(features_1)
    del features_1

    features_2, _ = clf.features_and_preds(NpzStreamer(args.batch_2))
    stats_2 = compute_statistics(features_2)
    del features_2

    print(f"P-FID: {stats_1.frechet_distance(stats_2)}")




action_id_to_intent = {v: k for k, v in ALL_INTENT.items()}

def normalize_point_clouds(pc: np.ndarray) -> np.ndarray:
    centroids = np.mean(pc, axis=1, keepdims=True)
    pc = pc - centroids
    m = np.max(np.sqrt(np.sum(pc**2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / m
    return pc

def features_and_preds_from_memory(
    self, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    data = normalize_point_clouds(data)

    output_features = []
    output_predictions = []

    batch_size = self.device_batch_size * len(self.devices)
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        batches = [
            torch.from_numpy(batch[j : j + self.device_batch_size])
            .permute(0, 2, 1) 
            .to(dtype=torch.float32, device=device)
            for j, device in zip(range(0, len(batch), self.device_batch_size), self.devices)
        ]

        for i, batch in enumerate(batches):
            with torch.no_grad():
                logits, _, features = self.models[i](batch, features=True)
                output_features.append(features.cpu().numpy())
                output_predictions.append(logits.exp().cpu().numpy())

    return np.concatenate(output_features, axis=0), np.concatenate(output_predictions, axis=0)

PointNetClassifier.features_and_preds_from_memory = features_and_preds_from_memory

def eval(eval_dir):

    grasps = pickle.load(open(os.path.join(eval_dir, 'res_diffuser.pkl'), 'rb'))
    gt = pickle.load(open(os.path.join("/home/syks/.cache/OakInkShape/1.2.1/", '231fe161b4fbce7448d50176b22f0b18.pkl'), 'rb'))
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
    # disjo_dist = AverageMeter("disjo_dist")
    sims_disp_sum = AverageMeter("sims_disp")
    sims_disp_var = []
    diversity = diversity_tester_wo_success(eval_dir)
    base_path = "/home/syks/Scene-Diffuser-obj/data/OakInk/shape"

    meta_path = os.path.join(base_path, "metaV2")
    real_meta = json.load(open(os.path.join(meta_path, "object_id.json"), "r"))
    virtual_meta = json.load(open(os.path.join(meta_path, "virtual_object_id.json"), "r"))
    num_coll = 0
    num_sum = 0

    min_fid_list = []
    clf = PointNetClassifier(devices=get_torch_devices())
    ksample = 20
    pbar = tqdm(total=len(grasps['sample_qpos']))



    for key, grasps_list in grasps['sample_qpos'].items():
        parts = key.split('-')
        name = parts[0]
        intent = parts[1]
        id_ = parts[2]
        random_index = random.randint(0, len(grasps_list) - 1)

        min_fid = float('inf')
        idx = random.random()
        for idx in range(len(gt['grasp_list'])):
            act_id = gt['grasp_list'][idx]["action_id"]
            intent_name = action_id_to_intent[act_id]
            oid = gt['grasp_list'][idx]["obj_id"]
            if id_ == oid and intent ==intent_name:
                gt_hand_pose = gt['grasp_list'][idx]['hand_pose']
                gt_hand_shape = gt['grasp_list'][idx]['hand_shape']
                gt_hand_transl_obj = gt['grasp_list'][idx]['hand_tsl']
                gt_verts = mano_layer(torch.from_numpy(gt_hand_pose).unsqueeze(0).float(),torch.from_numpy(gt_hand_shape).unsqueeze(0).float()).verts.squeeze(0).numpy()
                gt_verts = gt_verts + gt_hand_transl_obj
                gt_verts_batch = np.expand_dims(gt_verts, axis=0)
                hand_pose = grasps_list[random_index, :48]
                hand_shape = grasps_list[random_index, 48:58]
                hand_transl_obj = grasps_list[random_index, 58:]
                out_verts = mano_layer(torch.from_numpy(hand_pose).unsqueeze(0).float(),torch.from_numpy(hand_shape).unsqueeze(0).float()).verts.squeeze(0).numpy()
                out_verts = out_verts + hand_transl_obj

                out_verts_batch = np.expand_dims(out_verts, axis=0)
                features_1, _ = clf.features_and_preds_from_memory(out_verts_batch)
                stats_1 = compute_statistics(features_1)
                del features_1

                features_2, _ = clf.features_and_preds_from_memory(gt_verts_batch)
                stats_2 = compute_statistics(features_2)
                del features_2
                
                fid = stats_1.frechet_distance(stats_2)
                # print(f"P-FID: {fid}")
                if fid < min_fid:
                    min_fid = fid
            else:
                continue
        min_fid_list.append(min_fid)
        print(f"key is {key}, this min_fid is:{min_fid} ")
        print("--------------------------------------------------------------------------------------------")
        pbar.update(1)
    mean_min_fid = np.mean(min_fid_list)
    print(f"Mean Minimum FID: {mean_min_fid}")
    output_file = os.path.join(eval_dir, "PFID_new_quick.txt")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"mean_min_fid  is {mean_min_fid}")

        
    return 


def main() -> None:
    args = parse_args()
    eval(args.eval_dir)
    logger.info('End evaluating..')


if __name__ == '__main__':
    main()
