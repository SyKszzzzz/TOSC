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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def load_mesh_with_affordance(obj_path):
    """
    Load a mesh from an .obj file and extract affordance information.
    
    Parameters
    ----------
    obj_path: str
        Path to the .obj file.
    
    Returns
    -------
    mesh: trimesh.Trimesh
        The loaded 3D mesh.
    vertices: np.ndarray
        Vertices with the additional affordance as the fourth dimension.
    """
    # Load the mesh using trimesh
    mesh = tm.load(obj_path, force="mesh", process=False)
    vertices = []

    with open(obj_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                affordance = int(parts[4]) if len(parts) > 4 else 0
                vertices.append([x, y, z, affordance])
    
    vertices = np.array(vertices)
    
    return mesh, vertices

def get_ext_int(json_file_path):
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        object_mesh = data['rhand_mesh']
        dofValues = data['dofs']  # intrinsic
        trans = data['rhand_trans']
        quat = data['rhand_quat']

        

    #     extrinsic = dofs[0:7]# transform info of palm 1*7
    #     intrinsic = dofs[7:len(dofs)]

    # return extrinsic, intrinsic
    return dofValues, trans, quat

# 这个地方记得加回来！！！！！！！！！！！！！！！！！！！！
@DATASET.register()
class MultiDexManoUR(Dataset):
    """ Dataset for pose generation, training with MultiDex Dataset
    """

    _train_split = ['bag_8383', 'bag_8408', 'bag_8430', 'bag_8571', 'bag_8673', 'bag_8694', 'bag_8713', 'bag_8714', 'bag_8869', 'bag_8878', 'bag_8880', 'bag_8889', 
                 'bag_8899', 'bag_8907', 'bag_8908',    'bag_8920', 'bag_8929', 'bag_8932', 'bag_8952', 'bag_8953', 'bag_9047', 'bag_9056', 'bag_9058', 'bag_9066', 
                 'bag_9077', 'bag_9078', 'bag_9081', 'bag_9085', 'bag_9086', 'bag_9096', 'bag_9123', 'bag_9125', 'bag_9133', 'bag_9138', 'bag_9176', 'bag_9182', 
                 'bag_9184', 'bag_9206', 'bag_9210', 'bag_9229', 'bag_9231', 'bag_9233', 'bag_9238', 'bag_9244', 'bag_9300', 'bag_9315', 'bag_9317', 'bag_9347', 
                 'bag_9360', 'bag_9413', 'bag_9429', 'bag_9432', 'bag_9446', 
                 'bottle_3415', 'bottle_3418', 'bottle_3422', 'bottle_3443', 'bottle_3451', 'bottle_3486', 'bottle_3499', 'bottle_3502', 'bottle_3505', 'bottle_3506', 
                 'bottle_3508', 'bottle_3511', 'bottle_3512', 'bottle_3514', 'bottle_3515', 'bottle_3516', 'bottle_3517', 'bottle_3520', 'bottle_3528', 'bottle_3531', 
                 'bottle_3533', 'bottle_3534', 'bottle_3535', 'bottle_3542', 'bottle_3543', 'bottle_3547', 'bottle_3549', 'bottle_3552', 'bottle_3554', 'bottle_3558', 
                 'bottle_3561', 'bottle_3571', 'bottle_3574', 'bottle_3577', 'bottle_3578', 'bottle_3580', 'bottle_3584', 'bottle_3588', 'bottle_3593', 'bottle_3594', 
                 'bottle_3597', 'bottle_3599', 'bottle_3601', 'bottle_3603', 'bottle_3604', 'bottle_3606', 'bottle_3611', 'bottle_3612', 'bottle_3615', 'bottle_3617', 'bottle_3619', 'bottle_3625', 
                 'dispenser_101442', 'dispenser_101458', 'dispenser_101501', 'dispenser_101507', 'dispenser_101517', 'dispenser_101533', 'dispenser_101541', 
                 'dispenser_101557', 'dispenser_101560', 'dispenser_101561', 'dispenser_101563', 'dispenser_103355', 'dispenser_103378', 'dispenser_103380', 
                 'dispenser_103394', 'dispenser_103397', 'dispenser_103406', 'dispenser_103408', 'dispenser_103410', 'dispenser_103416', 'dispenser_103422', 
                 'dispenser_3398', 'dispenser_3410', 'dispenser_3586', 'dispenser_3632', 'dispenser_3686', 'dispenser_3736', 'dispenser_3749', 'dispenser_3776', 
                 'dispenser_3976', 'dispenser_3993', 'dispenser_4001', 'dispenser_4032', 'dispenser_4068', 
                 'faucet_1007', 'faucet_1011', 'faucet_1016', 'faucet_1034', 'faucet_1050', 'faucet_1052', 'faucet_1053', 'faucet_1280', 'faucet_1281', 'faucet_1285', 'faucet_1286', 
                 'faucet_1288', 'faucet_1292', 'faucet_1294', 'faucet_1296', 'faucet_1300', 'faucet_1304', 'faucet_1337', 'faucet_1380', 'faucet_1383', 'faucet_1384', 'faucet_1386', 
                 'faucet_1390', 'faucet_1392', 'faucet_1396', 'faucet_1459', 'faucet_1465', 'faucet_148', 'faucet_149', 'faucet_152', 'faucet_153', 'faucet_154', 'faucet_1548', 
                 'faucet_1551', 'faucet_156', 'faucet_165', 'faucet_167', 'faucet_168', 'faucet_170', 'faucet_229', 'faucet_295', 'faucet_383', 'faucet_584', 'faucet_677', 'faucet_693', 
                 'faucet_700', 'faucet_702', 'faucet_707', 'faucet_709', 'faucet_912', 'faucet_920', 'faucet_929', 'faucet_931', 'faucet_988', 'faucet_991', 
                 'handle-bottle_3381', 'handle-bottle_3408', 'handle-bottle_3425', 'handle-bottle_3570', 'handle-bottle_3579', 'handle-bottle_3591', 'handle-bottle_3592', 
                 'handle-bottle_3640', 'handle-bottle_3646', 'handle-bottle_3660', 'handle-bottle_3680', 'handle-bottle_3681', 'handle-bottle_3695', 'handle-bottle_3707', 
                 'handle-bottle_3710', 'handle-bottle_3718', 'handle-bottle_3726', 'handle-bottle_3745', 'handle-bottle_3752', 'handle-bottle_3812', 'handle-bottle_3849', 
                 'handle-bottle_3863', 'handle-bottle_3876', 'handle-bottle_3922', 'handle-bottle_3945', 'handle-bottle_3971', 'handle-bottle_4052', 'handle-bottle_4054', 
                 'handle-bottle_4061', 'handle-bottle_4071', 'handle-bottle_4074', 'handle-bottle_4918', 
                 'keyboard_12686', 'keyboard_12688', 'keyboard_12689', 'keyboard_12696', 'keyboard_12697', 'keyboard_12700', 'keyboard_12701', 'keyboard_12705', 'keyboard_12706', 
                 'keyboard_12714', 'keyboard_12715', 'keyboard_12716', 'keyboard_12727', 'keyboard_12738', 'keyboard_12819', 'keyboard_12827', 'keyboard_12828', 'keyboard_12829', 
                 'keyboard_12834', 'keyboard_12836', 'keyboard_12837', 'keyboard_12838', 'keyboard_12839', 'keyboard_12840', 'keyboard_12843', 'keyboard_12848', 'keyboard_12857', 
                 'keyboard_12860', 'keyboard_12861', 'keyboard_12875', 'keyboard_12876', 'keyboard_12880', 'keyboard_12884', 'keyboard_7388', 'keyboard_7390', 'keyboard_7393', 
                 'keyboard_7394', 'keyboard_7614', 'keyboard_7619', 'keyboard_7620', 'keyboard_7676', 'keyboard_7677', 'keyboard_7678', 'keyboard_7682', 'keyboard_7684', 'keyboard_7687', 
                 'keyboard_7690', 'keyboard_7692', 'keyboard_7695', 'keyboard_7698', 'keyboard_7699', 'keyboard_7700', 'keyboard_7701', 
                 'knife_1000', 'knife_1001', 'knife_1002', 'knife_1003', 'knife_1006', 'knife_1010', 'knife_1012', 'knife_1015', 'knife_1017', 'knife_1020', 'knife_1021', 
                 'knife_1024', 'knife_1025', 'knife_1027', 'knife_1030', 'knife_1031', 'knife_1032', 'knife_1033', 'knife_1038', 'knife_1042', 'knife_1046', 'knife_1049', 
                 'knife_1054', 'knife_1058', 'knife_1062', 'knife_1067', 'knife_1070', 'knife_1071', 'knife_1074', 'knife_1075', 'knife_1079', 'knife_1081', 'knife_1082', 
                 'knife_1083', 'knife_1085', 'knife_1088', 'knife_1089', 'knife_114', 'knife_115', 'knife_116', 'knife_117', 'knife_118', 'knife_132', 'knife_133', 'knife_145', 
                 'knife_146', 'knife_207', 'knife_218', 'knife_220', 'knife_221', 'knife_223', 'knife_225', 'knife_226', 'knife_230', 'knife_231', 'knife_42', 'knife_97', 
                 'laptop_10005', 'laptop_10012', 'laptop_10023', 'laptop_10025', 'laptop_10035', 'laptop_10037', 'laptop_10040', 'laptop_10046', 'laptop_10054', 'laptop_10070', 
                 'laptop_10071', 'laptop_10072', 'laptop_10076', 'laptop_10077', 'laptop_10081', 'laptop_10090', 'laptop_10092', 'laptop_10098', 'laptop_10101', 'laptop_10108', 
                 'laptop_10109', 'laptop_10114', 'laptop_10124', 'laptop_10125', 'laptop_10128', 'laptop_10133', 'laptop_10152', 'laptop_10153', 'laptop_10154', 'laptop_10156', 
                 'laptop_9604', 'laptop_9682', 'laptop_9685', 'laptop_9697', 'laptop_9703', 'laptop_9704', 'laptop_9705', 'laptop_9707', 'laptop_9708', 'laptop_9709', 'laptop_9713', 
                 'laptop_9714', 'laptop_9715', 'laptop_9726', 'laptop_9736', 'laptop_9738', 'laptop_9745', 'laptop_9746', 'laptop_9747', 'laptop_9748', 
                 'mug_8554', 'mug_8555', 'mug_8556', 'mug_8558', 'mug_8560', 'mug_8562', 'mug_8563', 'mug_8564', 'mug_8566', 'mug_8567', 'mug_8568', 'mug_8572', 'mug_8573', 
                 'mug_8574', 'mug_8575', 'mug_8576', 'mug_8578', 'mug_8579', 'mug_8580', 'mug_8581', 'mug_8582', 'mug_8588', 'mug_8591', 'mug_8594', 'mug_8595', 'mug_8608', 
                 'mug_8609', 'mug_8614', 'mug_8619', 'mug_8623', 'mug_8624', 'mug_8625', 'mug_8626', 'mug_8629', 'mug_8633', 'mug_8636', 'mug_8637', 'mug_8639', 'mug_8642', 
                 'mug_8645', 'mug_8646', 'mug_8647', 'mug_8649', 'mug_8651', 'mug_8656', 'mug_8658', 'mug_8659', 'mug_8661', 'mug_8663', 'mug_8664', 'mug_8666', 'mug_8756', 
                 'mug_8758', 'mug_8760', 'mug_8763', 
                 'scissors_10412', 'scissors_10429', 'scissors_10431', 'scissors_10435', 'scissors_10444', 'scissors_10449', 'scissors_10450', 'scissors_10471', 'scissors_10483', 
                 'scissors_10485', 'scissors_10487', 'scissors_10495', 'scissors_10499', 'scissors_10502', 'scissors_10508', 'scissors_10514', 'scissors_10516', 'scissors_10517', 
                 'scissors_10519', 'scissors_10528', 'scissors_10535', 'scissors_10537', 'scissors_10538', 'scissors_10540', 'scissors_10544', 'scissors_10548', 'scissors_10557', 
                 'scissors_10558', 'scissors_10559', 'scissors_10564', 'scissors_10570', 'scissors_10571', 'scissors_10577', 'scissors_10786', 'scissors_10787', 'scissors_10836', 
                 'scissors_10842', 'scissors_10884', 'scissors_10889', 'scissors_10891', 'scissors_10906', 'scissors_10913', 'scissors_10928', 'scissors_10933', 'scissors_10952', 
                 'scissors_10960', 'scissors_10962', 'scissors_10968', 'scissors_10973', 'scissors_10975', 'scissors_10994', 'scissors_11000', 'scissors_11020', 'scissors_11029', 
                 'scissors_11036', 'scissors_11047', 'scissors_11052']
    
    _test_split = ['jar_4118', 'jar_4197', 'jar_4200', 'jar_4216', 'jar_4233', 'jar_4314', 'jar_4343', 'jar_4403', 'jar_4409', 'jar_4427', 'jar_4514', 'jar_5619', 'jar_5675', 
                 'jar_5781', 'jar_5813', 'jar_5850', 'jar_5861', 'jar_5878', 'jar_5884', 'jar_5902', 'jar_5904', 'jar_5910', 'jar_5935', 'jar_5937', 'jar_5945', 'jar_5949', 
                 'jar_5951', 'jar_5955', 'jar_6004', 'jar_6008', 'jar_6017', 'jar_6037', 'jar_6040', 'jar_6077', 'jar_6120', 'jar_6193', 'jar_6217', 'jar_6222', 'jar_6335', 
                 'jar_6415', 'jar_6428', 'jar_6430', 'jar_6468', 'jar_6480', 'jar_6493', 
                 'pot_4115', 'pot_4121', 'pot_4131', 'pot_4139', 'pot_4156', 'pot_4183', 'pot_4187', 'pot_4188', 'pot_4189', 'pot_4217', 'pot_4218', 'pot_4220', 'pot_4227', 
                 'pot_4242', 'pot_4255', 'pot_4257', 'pot_4259', 'pot_4262', 'pot_4297', 'pot_4309', 'pot_4316', 'pot_4340', 'pot_4345', 'pot_4379', 'pot_4384', 'pot_4386', 
                 'pot_4387', 'pot_4414', 'pot_4416', 'pot_4429', 'pot_4432', 'pot_4434', 'pot_4457', 'pot_4465', 'pot_4471', 'pot_4479', 'pot_4480', 'pot_4481', 'pot_4501', 
                 'pot_5644', 'pot_5719', 'pot_5770', 'pot_5790', 'pot_5829', 'pot_5857', 'pot_5896', 'pot_6055', 'pot_6093', 
                 'earphone_10000', 'earphone_10003', 'earphone_10034', 'earphone_10112', 'earphone_10193', 'earphone_10506', 'earphone_10511', 'earphone_10549', 
                 'earphone_10551', 'earphone_10554', 'earphone_10565', 'earphone_10574', 'earphone_10621', 'earphone_10623', 'earphone_10643', 'earphone_10656', 
                 'earphone_10667', 'earphone_10677', 'earphone_10708', 'earphone_10709', 'earphone_10725', 'earphone_10750', 'earphone_10818', 'earphone_10846', 
                 'earphone_10871', 'earphone_10881', 'earphone_10887', 'earphone_10903', 'earphone_10914', 'earphone_10966', 'earphone_10970', 'earphone_11025', 
                 'earphone_11050', 'earphone_11057', 'earphone_11078', 'earphone_9657', 'earphone_9694', 'earphone_9777', 'earphone_9789', 'earphone_9807', 'earphone_9823', 
                 'earphone_9896', 'earphone_9913', 'earphone_9921', 'earphone_9927', 'earphone_9935', 'earphone_9982', 'earphone_9990', 'earphone_9998', 'earphone_9999']

    _all_split =['bag_8383', 'bag_8408', 'bag_8430', 'bag_8571', 'bag_8673', 'bag_8694', 'bag_8713', 'bag_8714', 'bag_8869', 'bag_8878', 'bag_8880', 'bag_8889', 
                 'bag_8899', 'bag_8907', 'bag_8908',    'bag_8920', 'bag_8929', 'bag_8932', 'bag_8952', 'bag_8953', 'bag_9047', 'bag_9056', 'bag_9058', 'bag_9066', 
                 'bag_9077', 'bag_9078', 'bag_9081', 'bag_9085', 'bag_9086', 'bag_9096', 'bag_9123', 'bag_9125', 'bag_9133', 'bag_9138', 'bag_9176', 'bag_9182', 
                 'bag_9184', 'bag_9206', 'bag_9210', 'bag_9229', 'bag_9231', 'bag_9233', 'bag_9238', 'bag_9244', 'bag_9300', 'bag_9315', 'bag_9317', 'bag_9347', 
                 'bag_9360', 'bag_9413', 'bag_9429', 'bag_9432', 'bag_9446', 
                 'bottle_3415', 'bottle_3418', 'bottle_3422', 'bottle_3443', 'bottle_3451', 'bottle_3486', 'bottle_3499', 'bottle_3502', 'bottle_3505', 'bottle_3506', 
                 'bottle_3508', 'bottle_3511', 'bottle_3512', 'bottle_3514', 'bottle_3515', 'bottle_3516', 'bottle_3517', 'bottle_3520', 'bottle_3528', 'bottle_3531', 
                 'bottle_3533', 'bottle_3534', 'bottle_3535', 'bottle_3542', 'bottle_3543', 'bottle_3547', 'bottle_3549', 'bottle_3552', 'bottle_3554', 'bottle_3558', 
                 'bottle_3561', 'bottle_3571', 'bottle_3574', 'bottle_3577', 'bottle_3578', 'bottle_3580', 'bottle_3584', 'bottle_3588', 'bottle_3593', 'bottle_3594', 
                 'bottle_3597', 'bottle_3599', 'bottle_3601', 'bottle_3603', 'bottle_3604', 'bottle_3606', 'bottle_3611', 'bottle_3612', 'bottle_3615', 'bottle_3617', 'bottle_3619', 'bottle_3625', 
                 'dispenser_101442', 'dispenser_101458', 'dispenser_101501', 'dispenser_101507', 'dispenser_101517', 'dispenser_101533', 'dispenser_101541', 
                 'dispenser_101557', 'dispenser_101560', 'dispenser_101561', 'dispenser_101563', 'dispenser_103355', 'dispenser_103378', 'dispenser_103380', 
                 'dispenser_103394', 'dispenser_103397', 'dispenser_103406', 'dispenser_103408', 'dispenser_103410', 'dispenser_103416', 'dispenser_103422', 
                 'dispenser_3398', 'dispenser_3410', 'dispenser_3586', 'dispenser_3632', 'dispenser_3686', 'dispenser_3736', 'dispenser_3749', 'dispenser_3776', 
                 'dispenser_3976', 'dispenser_3993', 'dispenser_4001', 'dispenser_4032', 'dispenser_4068', 
                 'earphone_10000', 'earphone_10003', 'earphone_10034', 'earphone_10112', 'earphone_10193', 'earphone_10506', 'earphone_10511', 'earphone_10549', 
                 'earphone_10551', 'earphone_10554', 'earphone_10565', 'earphone_10574', 'earphone_10621', 'earphone_10623', 'earphone_10643', 'earphone_10656', 
                 'earphone_10667', 'earphone_10677', 'earphone_10708', 'earphone_10709', 'earphone_10725', 'earphone_10750', 'earphone_10818', 'earphone_10846', 
                 'earphone_10871', 'earphone_10881', 'earphone_10887', 'earphone_10903', 'earphone_10914', 'earphone_10966', 'earphone_10970', 'earphone_11025', 
                 'earphone_11050', 'earphone_11057', 'earphone_11078', 'earphone_9657', 'earphone_9694', 'earphone_9777', 'earphone_9789', 'earphone_9807', 'earphone_9823', 
                 'earphone_9896', 'earphone_9913', 'earphone_9921', 'earphone_9927', 'earphone_9935', 'earphone_9982', 'earphone_9990', 'earphone_9998', 'earphone_9999', 
                 'faucet_1007', 'faucet_1011', 'faucet_1016', 'faucet_1034', 'faucet_1050', 'faucet_1052', 'faucet_1053', 'faucet_1280', 'faucet_1281', 'faucet_1285', 'faucet_1286', 
                 'faucet_1288', 'faucet_1292', 'faucet_1294', 'faucet_1296', 'faucet_1300', 'faucet_1304', 'faucet_1337', 'faucet_1380', 'faucet_1383', 'faucet_1384', 'faucet_1386', 
                 'faucet_1390', 'faucet_1392', 'faucet_1396', 'faucet_1459', 'faucet_1465', 'faucet_148', 'faucet_149', 'faucet_152', 'faucet_153', 'faucet_154', 'faucet_1548', 
                 'faucet_1551', 'faucet_156', 'faucet_165', 'faucet_167', 'faucet_168', 'faucet_170', 'faucet_229', 'faucet_295', 'faucet_383', 'faucet_584', 'faucet_677', 'faucet_693', 
                 'faucet_700', 'faucet_702', 'faucet_707', 'faucet_709', 'faucet_912', 'faucet_920', 'faucet_929', 'faucet_931', 'faucet_988', 'faucet_991', 
                 'handle-bottle_3381', 'handle-bottle_3408', 'handle-bottle_3425', 'handle-bottle_3570', 'handle-bottle_3579', 'handle-bottle_3591', 'handle-bottle_3592', 
                 'handle-bottle_3640', 'handle-bottle_3646', 'handle-bottle_3660', 'handle-bottle_3680', 'handle-bottle_3681', 'handle-bottle_3695', 'handle-bottle_3707', 
                 'handle-bottle_3710', 'handle-bottle_3718', 'handle-bottle_3726', 'handle-bottle_3745', 'handle-bottle_3752', 'handle-bottle_3812', 'handle-bottle_3849', 
                 'handle-bottle_3863', 'handle-bottle_3876', 'handle-bottle_3922', 'handle-bottle_3945', 'handle-bottle_3971', 'handle-bottle_4052', 'handle-bottle_4054', 
                 'handle-bottle_4061', 'handle-bottle_4071', 'handle-bottle_4074', 'handle-bottle_4918', 
                 'jar_4118', 'jar_4197', 'jar_4200', 'jar_4216', 'jar_4233', 'jar_4314', 'jar_4343', 'jar_4403', 'jar_4409', 'jar_4427', 'jar_4514', 'jar_5619', 'jar_5675', 
                 'jar_5781', 'jar_5813', 'jar_5850', 'jar_5861', 'jar_5878', 'jar_5884', 'jar_5902', 'jar_5904', 'jar_5910', 'jar_5935', 'jar_5937', 'jar_5945', 'jar_5949', 
                 'jar_5951', 'jar_5955', 'jar_6004', 'jar_6008', 'jar_6017', 'jar_6037', 'jar_6040', 'jar_6077', 'jar_6120', 'jar_6193', 'jar_6217', 'jar_6222', 'jar_6335', 
                 'jar_6415', 'jar_6428', 'jar_6430', 'jar_6468', 'jar_6480', 'jar_6493', 
                 'keyboard_12686', 'keyboard_12688', 'keyboard_12689', 'keyboard_12696', 'keyboard_12697', 'keyboard_12700', 'keyboard_12701', 'keyboard_12705', 'keyboard_12706', 
                 'keyboard_12714', 'keyboard_12715', 'keyboard_12716', 'keyboard_12727', 'keyboard_12738', 'keyboard_12819', 'keyboard_12827', 'keyboard_12828', 'keyboard_12829', 
                 'keyboard_12834', 'keyboard_12836', 'keyboard_12837', 'keyboard_12838', 'keyboard_12839', 'keyboard_12840', 'keyboard_12843', 'keyboard_12848', 'keyboard_12857', 
                 'keyboard_12860', 'keyboard_12861', 'keyboard_12875', 'keyboard_12876', 'keyboard_12880', 'keyboard_12884', 'keyboard_7388', 'keyboard_7390', 'keyboard_7393', 
                 'keyboard_7394', 'keyboard_7614', 'keyboard_7619', 'keyboard_7620', 'keyboard_7676', 'keyboard_7677', 'keyboard_7678', 'keyboard_7682', 'keyboard_7684', 'keyboard_7687', 
                 'keyboard_7690', 'keyboard_7692', 'keyboard_7695', 'keyboard_7698', 'keyboard_7699', 'keyboard_7700', 'keyboard_7701', 
                 'knife_1000', 'knife_1001', 'knife_1002', 'knife_1003', 'knife_1006', 'knife_1010', 'knife_1012', 'knife_1015', 'knife_1017', 'knife_1020', 'knife_1021', 
                 'knife_1024', 'knife_1025', 'knife_1027', 'knife_1030', 'knife_1031', 'knife_1032', 'knife_1033', 'knife_1038', 'knife_1042', 'knife_1046', 'knife_1049', 
                 'knife_1054', 'knife_1058', 'knife_1062', 'knife_1067', 'knife_1070', 'knife_1071', 'knife_1074', 'knife_1075', 'knife_1079', 'knife_1081', 'knife_1082', 
                 'knife_1083', 'knife_1085', 'knife_1088', 'knife_1089', 'knife_114', 'knife_115', 'knife_116', 'knife_117', 'knife_118', 'knife_132', 'knife_133', 'knife_145', 
                 'knife_146', 'knife_207', 'knife_218', 'knife_220', 'knife_221', 'knife_223', 'knife_225', 'knife_226', 'knife_230', 'knife_231', 'knife_42', 'knife_97', 
                 'laptop_10005', 'laptop_10012', 'laptop_10023', 'laptop_10025', 'laptop_10035', 'laptop_10037', 'laptop_10040', 'laptop_10046', 'laptop_10054', 'laptop_10070', 
                 'laptop_10071', 'laptop_10072', 'laptop_10076', 'laptop_10077', 'laptop_10081', 'laptop_10090', 'laptop_10092', 'laptop_10098', 'laptop_10101', 'laptop_10108', 
                 'laptop_10109', 'laptop_10114', 'laptop_10124', 'laptop_10125', 'laptop_10128', 'laptop_10133', 'laptop_10152', 'laptop_10153', 'laptop_10154', 'laptop_10156', 
                 'laptop_9604', 'laptop_9682', 'laptop_9685', 'laptop_9697', 'laptop_9703', 'laptop_9704', 'laptop_9705', 'laptop_9707', 'laptop_9708', 'laptop_9709', 'laptop_9713', 
                 'laptop_9714', 'laptop_9715', 'laptop_9726', 'laptop_9736', 'laptop_9738', 'laptop_9745', 'laptop_9746', 'laptop_9747', 'laptop_9748', 
                 'mug_8554', 'mug_8555', 'mug_8556', 'mug_8558', 'mug_8560', 'mug_8562', 'mug_8563', 'mug_8564', 'mug_8566', 'mug_8567', 'mug_8568', 'mug_8572', 'mug_8573', 
                 'mug_8574', 'mug_8575', 'mug_8576', 'mug_8578', 'mug_8579', 'mug_8580', 'mug_8581', 'mug_8582', 'mug_8588', 'mug_8591', 'mug_8594', 'mug_8595', 'mug_8608', 
                 'mug_8609', 'mug_8614', 'mug_8619', 'mug_8623', 'mug_8624', 'mug_8625', 'mug_8626', 'mug_8629', 'mug_8633', 'mug_8636', 'mug_8637', 'mug_8639', 'mug_8642', 
                 'mug_8645', 'mug_8646', 'mug_8647', 'mug_8649', 'mug_8651', 'mug_8656', 'mug_8658', 'mug_8659', 'mug_8661', 'mug_8663', 'mug_8664', 'mug_8666', 'mug_8756', 
                 'mug_8758', 'mug_8760', 'mug_8763', 
                 'pot_4115', 'pot_4121', 'pot_4131', 'pot_4139', 'pot_4156', 'pot_4183', 'pot_4187', 'pot_4188', 'pot_4189', 'pot_4217', 'pot_4218', 'pot_4220', 'pot_4227', 
                 'pot_4242', 'pot_4255', 'pot_4257', 'pot_4259', 'pot_4262', 'pot_4297', 'pot_4309', 'pot_4316', 'pot_4340', 'pot_4345', 'pot_4379', 'pot_4384', 'pot_4386', 
                 'pot_4387', 'pot_4414', 'pot_4416', 'pot_4429', 'pot_4432', 'pot_4434', 'pot_4457', 'pot_4465', 'pot_4471', 'pot_4479', 'pot_4480', 'pot_4481', 'pot_4501', 
                 'pot_5644', 'pot_5719', 'pot_5770', 'pot_5790', 'pot_5829', 'pot_5857', 'pot_5896', 'pot_6055', 'pot_6093', 
                 'scissors_10412', 'scissors_10429', 'scissors_10431', 'scissors_10435', 'scissors_10444', 'scissors_10449', 'scissors_10450', 'scissors_10471', 'scissors_10483', 
                 'scissors_10485', 'scissors_10487', 'scissors_10495', 'scissors_10499', 'scissors_10502', 'scissors_10508', 'scissors_10514', 'scissors_10516', 'scissors_10517', 
                 'scissors_10519', 'scissors_10528', 'scissors_10535', 'scissors_10537', 'scissors_10538', 'scissors_10540', 'scissors_10544', 'scissors_10548', 'scissors_10557', 
                 'scissors_10558', 'scissors_10559', 'scissors_10564', 'scissors_10570', 'scissors_10571', 'scissors_10577', 'scissors_10786', 'scissors_10787', 'scissors_10836', 
                 'scissors_10842', 'scissors_10884', 'scissors_10889', 'scissors_10891', 'scissors_10906', 'scissors_10913', 'scissors_10928', 'scissors_10933', 'scissors_10952', 
                 'scissors_10960', 'scissors_10962', 'scissors_10968', 'scissors_10973', 'scissors_10975', 'scissors_10994', 'scissors_11000', 'scissors_11020', 'scissors_11029', 
                 'scissors_11036', 'scissors_11047', 'scissors_11052']


    quat_joint_angle_lower = torch.tensor([-0.540872, -0.853893, -0.859689, -0.752555])
    quat_joint_angle_upper = torch.tensor([1.047816, 1.042326, 1.035418, 1.01422])
    

  
    

    dof_joint_angle_lower = torch.tensor([-0.359375, -0.18605, -0.181934, -0.185633, -0.185676, -0.185619, -0.359112, -0.186293, -0.186295, -0.531883,
                                       -0.185859, -0.186388, -0.635082, -1.13582, -0.011977, -0.011813])
    

    dof_joint_angle_upper = torch.tensor([0.360689, 1.580947, 1.582204, 0.18628, 1.582033, 1.582713, 0.349066,
                                       1.580642, 1.581296, 0.523599, 1.582212, 1.581999, 1.233036, 0.444647,
                                       1.581274, 1.755159])

    # _global_trans_lower = torch.tensor([-190.192, -250.759, -197.847])
    # _global_trans_upper = torch.tensor([205.25, 366.653, 214.061])

    _global_trans_lower = torch.tensor([-0.190192, -0.250759, -0.197847])
    _global_trans_upper = torch.tensor([0.205250, 0.366653, 0.214061])


    # _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
    #                                    -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
    #                                    -0.5237, 0.])
    # _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
    #                                    1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
    #                                    0.6981317, 0.43633232, 1.5707964,  1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
    #                                    0.5237, 1.])

    

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(MultiDexManoUR, self).__init__()
        self.phase = phase
        self.slurm = slurm
        if self.phase == 'train':
            self.split = self._train_split
        elif self.phase == 'test':
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.device = cfg.device
        self.is_downsample = cfg.is_downsample # True
        self.modeling_keys = cfg.modeling_keys # ['allDoFs']
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color  # false
        self.use_normal = cfg.use_normal # false
        self.normalize_x = cfg.normalize_x  # true
        self.normalize_x_trans = cfg.normalize_x_trans  # true
        self.obj_dim = int(3 + 3 * self.use_color + 3 * self.use_normal)
        self.transform = make_default_transform(cfg, phase)  # 这里的transform是numpy to tensor

        ## resource folders
        # self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        # self.data_dir = os.path.join(self.asset_dir, 'shadowhand')
        # self.scene_path = os.path.join(self.asset_dir, 'object_pcds.pkl') # 这里就是物体的点云，全部的，没有采样过的

        self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        self.data_dir = os.path.join(self.asset_dir, 'grasp')
        self.scene_path = os.path.join(self.asset_dir, 'object_afford') # 这里就是物体的点云，全部的，没有采样过的

        self.obj_gpt_dir = os.path.join(self.asset_dir, 'obj_gpt')
        self.task_gpt_dir = os.path.join(self.asset_dir, 'task_gpt')
        self.task_ins_dir = os.path.join(self.asset_dir, 'ins_gpt')


        self.quat_joint_angle_lower = self.quat_joint_angle_lower.cpu()
        self.quat_joint_angle_upper = self.quat_joint_angle_upper.cpu()
        self.dof_joint_angle_lower = self.dof_joint_angle_lower.cpu()
        self.dof_joint_angle_upper = self.dof_joint_angle_upper.cpu()
        self._global_trans_lower = self._global_trans_lower.cpu()
        self._global_trans_upper = self._global_trans_upper.cpu()

        ## load data
        self._pre_load_data(case_only)

    def _pre_load_data(self, case_only: bool) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.frames = []
        self.scene_pcds = {}
        self.scene_meshes = {}
        self.object_affordance_list = {}

        for s in self.split:
            object_name = s.split("_")[0]
            object_index = s.split("_")[1]

            obj_path = os.path.join(self.scene_path, s+".obj")
            # scene_mesh = trimesh.load(os.path.join(self.scene_path, s+".obj"),force="mesh", process=False)
            scene_mesh, vertices_afford = load_mesh_with_affordance(obj_path)
            self.scene_meshes[s] = scene_mesh

            affordances = np.zeros((vertices_afford.shape[0], 1), dtype=np.uint8)
            for idx, vertex in enumerate(vertices_afford):
                affordance = int(vertex[3])
                affordances[idx] = affordance
            self.object_affordance_list[s] = affordances

            vertices = torch.tensor(self.scene_meshes[s].vertices, dtype=torch.float, device=self.device)
            faces = torch.tensor(self.scene_meshes[s].faces, dtype=torch.float, device=self.device)
            mesh = Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
            dense_point_cloud = sample_points_from_meshes(mesh, num_samples=100 * self.num_points)
            # print("dense_point_cloud shape is ", dense_point_cloud.shape)
            # print("dense_point_cloud is ", dense_point_cloud)

            surface_points, sampled_indices = sample_farthest_points(dense_point_cloud, K=self.num_points)
            surface_points = surface_points.squeeze(0).to(dtype=torch.float, device=self.device)
           
            afford_labels = torch.tensor(affordances, dtype=torch.float, device=self.device)

            distances = torch.cdist(surface_points, vertices)
            nearest_vertex_indices = distances.argmin(dim=1)

            sampled_afford_labels = afford_labels[nearest_vertex_indices].to(dtype=torch.float, device=self.device)

            # print("sampled_afford_labels shape is ", sampled_afford_labels.shape)
            # print("sampled_afford_labels is ", sampled_afford_labels)

            # print("suface_points shape is ", surface_points.shape)
            # print("suface_points type is ", type(surface_points))

            # sampled_afford_labels = affordances[sampled_indices.cpu()].to(dtype=torch.float, device=self.device)

            surface_points_with_labels = torch.cat((surface_points, sampled_afford_labels), dim=1)
            # print("surface_points_with_labels shape is ", sampled_afford_labels.shape)
            # print("surface_points_with_labels is ", surface_points_with_labels)
            # surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
            # surface_points.to(dtype=float, device=self.device)
            self.scene_pcds[s] = surface_points_with_labels



        # self.frames就是把所有的全放进来，需要给的标签是

        # self.frames.append({'robot_name': 'mano',
        #                             'object_name': xxxx,
        #                             'afford_label': xxxx,  # 物体的旋转矩阵
        #                             'qpos': mdata_qpos})# qpos就是label

        for object_folder in os.listdir(self.data_dir):
            object_folder_path = os.path.join(self.data_dir, object_folder)
            if os.path.isdir(object_folder_path):
                for object_index_folder in os.listdir(object_folder_path):
                    object_index_folder_path = os.path.join(object_folder_path, object_index_folder)

                    object_name = f'{object_folder}_{object_index_folder}'

                    if object_name in self.split:
                        if os.path.isdir(object_index_folder_path):
                            for task_folder in os.listdir(object_index_folder_path):
                                task_folder_path = os.path.join(object_index_folder_path, task_folder)

                                afford_label = f"{task_folder}".split("_")[1]

                                if os.path.isdir(task_folder_path):
                                    for file_name in os.listdir(task_folder_path):
                                        if file_name.endswith('.json'):
                                            file_path = os.path.join(task_folder_path, file_name)
                                            dofValues, trans, quat = get_ext_int(file_path)

                                            dofValues_tensor = torch.tensor(dofValues, dtype=torch.float32).cpu()
                                            trans_tensor = torch.tensor(trans, dtype=torch.float32).cpu()
                                            # trans_tensor = torch.tensor(trans, dtype=torch.float32).cpu()/1000

                                            quat_tensor = torch.tensor(quat, dtype=torch.float32).cpu()
                                            if self.normalize_x: 

                                                quat_tensor = self.quat_angle_normalize(quat_tensor)
                                                dofValues_tensor = self.dof_angle_normalize(dofValues_tensor)
                                                
                                            if self.normalize_x_trans:
                                                trans_tensor = self.trans_normalize(trans_tensor)
                                            
                                            dofs = torch.cat((trans_tensor, quat_tensor, dofValues_tensor))
                                            
                                            # 这里加上一个标准化的过程
                                            # self.file_paths.append(file_path)
                                            self.frames.append({'robot_name': 'mano',
                                                                'object_name':object_name,  # name_index
                                                                'afford_label': afford_label,  # 物体的旋转矩阵
                                                                'qpos': dofs})# qpos就是label
                    else:
                        break
        print('Finishing Pre-load in MultiDexManoUR')

    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm

    def quat_angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self.quat_joint_angle_lower), (self.quat_joint_angle_upper - self.quat_joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return joint_angle_norm

    def quat_angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self.quat_joint_angle_upper - self.quat_joint_angle_lower) + self.quat_joint_angle_lower
        return joint_angle_denorm
    
    def dof_angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self.dof_joint_angle_lower), (self.dof_joint_angle_upper - self.dof_joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return joint_angle_norm

    def dof_angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self.dof_joint_angle_upper - self.dof_joint_angle_lower) + self.dof_joint_angle_lower
        return joint_angle_denorm

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index: Any) -> Tuple:

        frame = self.frames[index]

        ## load data, containing scene point cloud and point pose
        scene_id = frame['object_name']
        # scene_rot_mat = frame['object_rot_mat']
                # scene_mesh = self.scene_meshes[scene_id]
                # affordances = self.object_affordance_list[scene_id]


                # vertices = torch.tensor(scene_mesh.vertices, dtype=torch.float, device=self.device)
                # faces = torch.tensor(scene_mesh.faces, dtype=torch.float, device=self.device)

                # mesh = Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                # dense_point_cloud = sample_points_from_meshes(mesh, num_samples=100 * self.num_points)
                # print("dense_point_cloud shape is ", dense_point_cloud.shape)
                # print("dense_point_cloud is ", dense_point_cloud)

                # surface_points, sampled_indices = sample_farthest_points(dense_point_cloud, K=self.num_points)
                # surface_points = surface_points.squeeze(0).to(dtype=torch.float, device=self.device)
                
                # afford_labels = torch.tensor(affordances, dtype=torch.float, device=self.device)

                # distances = torch.cdist(surface_points, vertices)
                # nearest_vertex_indices = distances.argmin(dim=1)

                # sampled_afford_labels = afford_labels[nearest_vertex_indices].to(dtype=torch.float, device=self.device)

                # print("sampled_afford_labels shape is ", sampled_afford_labels.shape)
                # print("sampled_afford_labels is ", sampled_afford_labels)

                # print("suface_points shape is ", surface_points.shape)
                # print("suface_points type is ", type(surface_points))

                # # sampled_afford_labels = affordances[sampled_indices.cpu()].to(dtype=torch.float, device=self.device)

                # surface_points_with_labels = torch.cat((surface_points, sampled_afford_labels), dim=1)
                # print("surface_points_with_labels shape is ", surface_points_with_labels.shape)
                # print("surface_points_with_labels is ", surface_points_with_labels)
        # surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
        # surface_points.to(dtype=float, device=self.device)
        # self.scene_pcds[s] = surface_points_with_labels



        scene_pc = self.scene_pcds[scene_id]
        # scene_pc = np.einsum('mn, kn->km', scene_rot_mat, scene_pc) # 这里表示矩阵相乘
        cam_tran = None

        ## randomly resample points
        if self.phase != 'train':
            np.random.seed(0) # resample point cloud with a fixed random seed
        # np.random.shuffle(scene_pc)
        # scene_pc = scene_pc[:self.num_points]
        # resample_indices = np.random.permutation(len(scene_pc))
        # scene_pc = scene_pc[resample_indices[:self.num_points]]
        # 随机采样2048个点

        ## format point cloud xyz and feature
        xyz = scene_pc[:, 0:3]

        if self.use_color:
            feat = scene_pc[:, 3:4]
            # feat = np.concatenate([color], axis=-1)

        if self.use_normal:
            normal = scene_pc[:, 6:9]
            feat = np.concatenate([normal], axis=-1)

        ## format smplx parameters
        grasp_qpos = (
            frame['qpos']
        ) # 这个就是label，表征grasp的参数
        afford_lable = frame['afford_label']

        object_name = scene_id.split("_")[0]

        
        obj_desc_dir =  os.path.join(self.obj_gpt_dir, object_name, 'descriptions', str(np.random.randint(0, 10)))
        if not os.path.exists(obj_desc_dir):
            raise ValueError(f"No such object description dir: {obj_desc_dir}")
        obj_desc = np.load(os.path.join(obj_desc_dir, 'word_embed.npy'))[0]
        obj_desc_mask = np.load(os.path.join(obj_desc_dir, 'attn_mask.npy'))[0]

        # print("obj_desc type is ", type(obj_desc)," shape is ", obj_desc.shape," obj_desc_mask type is ", type(obj_desc_mask), " shape is ", obj_desc_mask.shape)
        # print("obj_desc is ", obj_desc," shape is ", obj_desc.shape," obj_desc_mask is ", obj_desc_mask, " shape is ", obj_desc_mask.shape)

               

        task_desc_dir = os.path.join(self.task_gpt_dir, afford_lable, 'descriptions', str(np.random.randint(0, 10)))
        if not os.path.exists(task_desc_dir):
            raise ValueError(f"No such task description dir: {task_desc_dir}")
        task_desc = np.load(os.path.join(task_desc_dir, 'word_embed.npy'))[0]
        task_desc_mask = np.load(os.path.join(task_desc_dir, 'attn_mask.npy'))[0]
        # print("task_desc is ", task_desc," shape is ", task_desc.shape," task_desc_mask is ", task_desc_mask, " shape is ", task_desc_mask.shape)
        # print("task_desc type is ", type(task_desc)," shape is ", task_desc.shape," task_desc_mask type is ", type(task_desc_mask), " shape is ", task_desc_mask.shape)


        task_ins_id = np.random.randint(0, 53)
        task_ins_path = os.path.join(self.task_ins_dir, object_name, afford_lable, str(task_ins_id)+'_word.npy')
        task_ins_mask_path = os.path.join(self.task_ins_dir, object_name, afford_lable, str(task_ins_id)+'_mask.npy')

        if not os.path.exists(task_ins_path) or not os.path.exists(task_ins_mask_path):
            raise ValueError(f"No such task instruction or mask file: {task_ins_path}")
        with open(task_ins_path, 'rb') as f:
            task_ins = np.load(f)[0]  # [21, 768]
        with open(task_ins_mask_path, 'rb') as f:
            task_ins_mask = np.load(f)[0]  # [21]

        # print("task_ins type is ", type(task_ins)," shape is ", task_ins.shape," task_ins_mask type is ", type(task_ins_mask), " shape is ", task_ins_mask.shape)
        

        data = {
            'x': grasp_qpos,  # Grasp参数，27维。AffordPse是23维
            'pos': xyz,  # 物体点云
            # 'feat':feat,  # 物体点云的特征
            # 'scene_rot_mat': scene_rot_mat, # 物体的旋转矩阵
            'obj_desc':obj_desc,
            'obj_desc_mask':obj_desc_mask,
            'task_desc':task_desc,
            'task_desc_mask':task_desc_mask,
            'task_ins':task_ins,
            'task_ins_mask':task_ins_mask,
            'afford_lable':afford_lable,
            'cam_tran': cam_tran,  # None
            'scene_id': scene_id,  # object_name+index
        }

        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys)

        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    config_path = "configs/task/grasp_gen_ur_mano.yaml"
    cfg = OmegaConf.load(config_path)
    dataloader = MultiDexManoUR(cfg.dataset, 'all', False).get_dataloader(batch_size=4,
                                                                                  collate_fn=collate_fn_squeeze_pcd_batch_grasp,
                                                                                  num_workers=0,
                                                                                  pin_memory=False,
                                                                                  shuffle=True,)
    global_min = None
    global_max = None

    device = 'cuda'
    for it, data in enumerate(dataloader):
        for key in data:
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(device)
            
            if key == 'x':
                print("x shape is ", data[key].shape)
                x_batch = data[key]
                batch_min = torch.min(x_batch, dim=0).values
                batch_max = torch.max(x_batch, dim=0).values
                    # 记录x中每一维度的最大值和最小值
                if global_min is None and global_max is None:
                        global_min = batch_min
                        global_max = batch_max
                else:
                    # 更新全局最小值和最大值
                    global_min = torch.min(global_min, batch_min)
                    global_max = torch.max(global_max, batch_max)

                    print(data[key].shape)
            print("global_min is ", global_min)
            print("global_max is ", global_max)
        print("next one !!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("global_min is ", global_min)
    print("global_max is ", global_max)
