from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.model.utils import timestep_embedding
from models.model.utils import ResBlock, SpatialTransformer
from models.model.scene_model import create_scene_model
from models.base import MODEL
from torch.nn import init
import math
import numpy as np
from torch.autograd import Variable
from models.model.clip_sd import ClipCustom

class ResBlockDexTOG(nn.Module):
    def __init__(self,
                 hidden_dim: int = 256,
                 temp_emb_cat: bool = False):
        super(ResBlockDexTOG, self).__init__()

        if temp_emb_cat:
            self.linear_layer = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        else:
            self.linear_layer = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        self.hidden_dim = hidden_dim
        self.temp_emb_cat = temp_emb_cat

    def forward(self, x, temp_emb, pcd_emb, text_emb):
        if self.temp_emb_cat:
            x_expand = torch.cat((x, temp_emb, pcd_emb, text_emb), dim=1)
        else:
            x_expand = torch.cat((x, pcd_emb, text_emb), dim=1) * temp_emb
        x = x + self.linear_layer(x_expand)
        return x

@MODEL.register()
class UNetModel(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super(UNetModel, self).__init__()

        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x

        ## create scene model from config
        self.scene_model_name = cfg.scene_model.name
        scene_model_in_dim = 3 + int(cfg.scene_model.use_color) * 3 + int(cfg.scene_model.use_normal) * 3
        if cfg.scene_model.name == 'PointNet':
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points,
                                'num_tokens': cfg.scene_model.num_tokens}
        else:
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points}
        self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)




        



        ## load pretrained weights
        weight_path = cfg.scene_model.pretrained_weights_slurm if slurm else cfg.scene_model.pretrained_weights
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)

        if cfg.freeze_scene_model:
            for p in self.scene_model.parameters():
                p.requires_grad_(False)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        # print("x_t.shape is ", x_t.shape)
        assert len(x_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)


        h = rearrange(x_t, 'b l c -> b c l')

        h = h.float()

        h = self.in_layers(h) # <B, d_model, L>

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def condition(self, data: Dict) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition feature
        """
        if self.scene_model_name == 'PointTransformer':
            b = data['offset'].shape[0]
            pos, feat, offset = data['pos'], data['feat'], data['offset']
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            scene_feat = rearrange(x5, '(b n) c -> b n c', b=b, n=self.scene_model.num_groups)
        elif self.scene_model_name == 'PointNet':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)
        elif self.scene_model_name == 'PointNet2':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            print("pos shape is ", pos.shape)
            _, scene_feat_list = self.scene_model(pos)
            scene_feat = scene_feat_list[-1].transpose(1, 2)
        elif self.scene_model_name == 'IDGC':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            print("pos shape is ", pos.shape)
            # data['ins']
            scene_feat = self.scene_model(pos, data['guidence'])
        else:
            raise Exception('Unexcepted scene model.')

        return scene_feat
    
class TimeEmbedding(nn.Module):

    def __init__(self, T: int, d_model: int, dim: int):
        super().__init__()
        emb = torch.arange(0, d_model, step = 2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim = -1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view((T, d_model))

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.time_embedding(t)
        return emb

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

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

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


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
    def __init__(self, out_dim = 256, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
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
        B, N, D = x.size()
        trans = self.stn(x.transpose(2, 1))
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

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
        x = x.view(-1, 1024)
        return x


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class PointNetEmb(nn.Module):
    def __init__(self, emb_dim = 256):
        super(PointNetEmb, self).__init__()
        self.encoder = PointNetEncoder(out_dim=emb_dim, channel=4)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x1, x2):
        ones = torch.ones((x1.shape[0], x1.shape[1], 1)).to(x1.device)
        zeros = torch.zeros((x2.shape[0], x2.shape[1], 1)).to(x2.device)
        x1 = torch.cat((x1, ones), dim=2)
        x2 = torch.cat((x2, zeros), dim=2)
        x = self.mlp(self.encoder(torch.cat((x1, x2), dim=1)))
        return x

@MODEL.register()
class DexTOGModel(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super(DexTOGModel, self).__init__()
        input_dim: int = 61
        hidden_dim: int = 256
        res_layers_num: int = 4
        diffusion_step = 100
        self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
        )
        temp_emb_cat = False
        self.temp_emb = TimeEmbedding(diffusion_step, hidden_dim // 4, hidden_dim * 3)
        self.linears = nn.ModuleList([ResBlockDexTOG(hidden_dim, temp_emb_cat=temp_emb_cat) for _ in range(res_layers_num)])
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.pcd_emb = PointNetEmb(emb_dim=hidden_dim)

        self.text_compress = nn.Sequential(
                nn.Linear(1536, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.temp_emb_cat = temp_emb_cat
        self.initialize()
        self.lr = 1e-3
        self.language_model = ClipCustom(cfg=None,num=1536)
        self.language_model.freeze()



        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x

        ## create scene model from config
        self.scene_model_name = cfg.scene_model.name
        scene_model_in_dim = 3 + int(cfg.scene_model.use_color) * 3 + int(cfg.scene_model.use_normal) * 3
        if cfg.scene_model.name == 'PointNet':
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points,
                                'num_tokens': cfg.scene_model.num_tokens}
        else:
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points}
        self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)

        ## load pretrained weights
        weight_path = cfg.scene_model.pretrained_weights_slurm if slurm else cfg.scene_model.pretrained_weights
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)
        if cfg.freeze_scene_model:
            for p in self.scene_model.parameters():
                p.requires_grad_(False)

    def initialize(self):
        for module in self.linears:
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight)
                init.zeros_(module.bias)
        for module in self.input_layer:
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight)
                init.zeros_(module.bias)
        init.kaiming_uniform_(self.output_layer.weight)
        init.zeros_(self.output_layer.bias)
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, obj_pcd, hand_pcd, guidence) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        t_emb = self.temp_emb(ts)
        # print("obj_pcd shape is ", obj_pcd, obj_pcd.shape, "hand_pcd shape is ", hand_pcd, hand_pcd.shape , "hand_pcd type is ", type(hand_pcd), "obj_pcd type is ", type(obj_pcd))
        pcd_emb = self.pcd_emb(obj_pcd.float(), hand_pcd.float())
        cond_txt, text_vector = self.language_model(guidence, None)
        cond_txt = cond_txt.mean(dim=1)
        # print(f"cond_txt shape is {cond_txt.shape}")
        text_compressed_emb = self.text_compress(cond_txt)
        x = x_t
        # print("x shape is ", x.shape)
        x = self.input_layer(x)

        # print(f"x shape is {x.shape}, t_emb shape is {t_emb.shape}, pcd_emb shape is {pcd_emb.shape}, text_compressed_emb shape is {text_compressed_emb.shape}")
        for idx, layer in enumerate(self.linears):
            x = layer(x, t_emb, pcd_emb, text_compressed_emb)
        pred_noise = self.output_layer(x)
        return pred_noise


    def condition(self, data: Dict, x_t_hand_verts_obj) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition feature
        """
        if self.scene_model_name == 'PointTransformer':
            b = data['offset'].shape[0]
            pos, feat, offset = data['pos'], data['feat'], data['offset']
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            scene_feat = rearrange(x5, '(b n) c -> b n c', b=b, n=self.scene_model.num_groups)
        elif self.scene_model_name == 'PointNet':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)

        elif self.scene_model_name == 'PointNet_SV':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'], data['obj_disc'], data['obj_mask_disc'])
            scene_feat = scene_feat_list.transpose(1, 2)

# def forward(self, pointcloud: torch.cuda.FloatTensor, ins_disc, ins_mask_disc, obj_disc, obj_mask_disc):

        elif self.scene_model_name == 'PointNet2':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            print("pos shape is ", pos.shape)
            _, scene_feat_list = self.scene_model(pos)
            scene_feat = scene_feat_list[-1].transpose(1, 2)



        elif self.scene_model_name == 'IDGC':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            print("pos shape is ", pos.shape)
            # data['ins']
            scene_feat = self.scene_model(pos, data['guidence'])
        
        elif self.scene_model_name == 'MAE':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            print("pos shape is ", pos.shape)
            # print("data['guidence'] is ",  data['guidence'])
            scene_feat = self.scene_model(pos, data['guidence'])

        elif self.scene_model_name == 'MAE_dual':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            pos_ori = torch.tensor(data['pos_ori']).cuda().to(torch.float32)
            print("pos shape is ", pos.shape)
            # print("data['guidence'] is ",  data['guidence'])
            scene_feat = self.scene_model(pos,pos_ori, data['guidence'])
            





        elif self.scene_model_name == 'PointNet2_LLM':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            obj_desc = data['obj_desc']
            obj_desc_mask = data['obj_desc_mask']
            task_desc = data['task_desc']
            task_desc_mask = data['task_desc_mask']
            task_ins = data['task_ins']
            task_ins_mask = data['task_ins_mask']
            # , task_desc, task_desc_mask, ins, ins_mask
            

            _, enriched_features = self.scene_model(pos,obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
            scene_feat = enriched_features.transpose(1, 2)

        elif self.scene_model_name == 'PointNet2_Part_wo_point':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            print("pos shape is ", pos.shape)
            print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2)

        elif self.scene_model_name == 'PointNet2_ADT':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2)
        elif self.scene_model_name == 'PointNet2_ADT_ins':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2)

        elif self.scene_model_name == 'PointNet2_ADT_full':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2)

        elif self.scene_model_name == 'PointNet2_ADT_all':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2)

        elif self.scene_model_name == 'PointNet2_ADT_ins_only':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2)

        elif self.scene_model_name == 'PointNet2_ADT_full_concat':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2) # （B，c_dim, 16）
        elif self.scene_model_name == 'ADT':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2) # （B，c_dim, 16）

        elif self.scene_model_name == 'PointNet2_ADT_full_concat_msg':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2) # （B，c_dim, 16）
        elif self.scene_model_name == 'PointNet2_ADT_point_only':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2) # （B，c_dim, 16）

        elif self.scene_model_name == 'PointNet2_ADT_l0_only':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2) # （B，c_dim, 16）

        elif self.scene_model_name == 'PointNet2_ADT_l1_only':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2) # （B，c_dim, 16）


        elif self.scene_model_name == 'PointNet2_ADT_pa_only':
            data['pos'] = torch.tensor(data['pos']).cuda()
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            # print("pos shape is ", pos.shape)
            # print("data['ins_disc'] shape is ", data['ins_disc'].shape) 

            _, scene_feat_list = self.scene_model(pos, data['ins_disc'], data['ins_mask_disc'],data['task_disc'],data['task_mask_disc'],data['obj_L0_part_disc'],data['obj_L0_part_disc_mask'],data['obj_L0_valid_part'], data['obj_L1_part_disc'],data['obj_L1_part_disc_mask'],data['obj_L1_valid_part'], data['obj_l0_part'], data['obj_l1_part'])
            scene_feat = scene_feat_list.transpose(1, 2) # （B，c_dim, 16）
        else:
            raise Exception('Unexcepted scene model.')

        return scene_feat
    
def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))

class TTA_PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(TTA_PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
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
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3 :
            x, feature = size_splits(x, [3, D-3], dim=2)#x.split(3,dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

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
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, condition_size=1024):
        super().__init__()

        if conditional:
            assert condition_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, condition_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, condition_size)

    def forward(self, x, c=None):
        # x: [B, 58]
        # if x.dim() > 2:
        #     x = x.view(-1, 58)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size], device=c.device)
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        #print('encoder', self.MLP)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1) # [B, 1024+61]
        #print('x size before MLP {}'.format(x.size()))
        x = self.MLP(x)
        #print('x size after MLP {}'.format(x.size()))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        #print('mean size {}, log_var size {}'.format(means.size(), log_vars.size()))
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        #print('decoder', self.MLP)

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)
            #print('z size {}'.format(z.size()))

        x = self.MLP(z)

        return x
    
