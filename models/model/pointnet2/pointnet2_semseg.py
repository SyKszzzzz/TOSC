import os
import sys

sys.path.append(os.path.join('.'))

import torch
import torch.nn as nn
from models.model.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG
import models.model.pointnet2.pytorch_utils as pt_utils
import math
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from models.model.clip_sd import ClipCustom
import time
# from model.clip_sd import ClipCustom
def get_model(num_classes, is_msg=True, input_channels=6, use_xyz=True, bn=True):
    if is_msg:
        model = Pointnet2MSG(
            num_classes=num_classes, 
            input_channels=input_channels, 
            use_xyz=use_xyz, 
            bn=bn
        )
    else:
        model = Pointnet2SSG(
        num_classes=num_classes, 
        input_channels=input_channels, 
        use_xyz=use_xyz, 
        bn=bn
    )

    return model

class Pointnet2MSG(nn.Module):
    def __init__(self, num_classes, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        NPOINTS = [1024, 256, 64, 16]
        RADIUS = [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
        NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
        MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]],
                [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
        FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
        CLS_FC = [128]
        DP_RATIO = 0.5

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k],
                    bn=bn
                )
            )

        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=bn))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, num_classes, activation=None, bn=bn))
        cls_layers.insert(1, nn.Dropout(DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, 1)
        return pred_cls

class Pointnet2SSG(nn.Module):
    def __init__(self, num_classes, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        NPOINTS = [1024, 256, 64, 16]
        RADIUS = [0.1, 0.2, 0.4, 0.8]
        NSAMPLE = [32, 32, 32, 32]
        MLPS = [[32, 32, 64], [64, 64, 128],
                [128, 128, 256], [256, 256, 512]]
        FP_MLPS = [[128, 128], [256, 128], [256, 256], [256, 256]]
        CLS_FC = [128]
        DP_RATIO = 0.5

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            mlps = [channel_in] + mlps
            channel_out += mlps[-1]

            self.SA_modules.append(
                PointnetSAModule(
                    npoint=NPOINTS[k],
                    radius=RADIUS[k],
                    nsample=NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k],
                    bn=bn
                )
            )

        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=bn))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, num_classes, activation=None, bn=bn))
        cls_layers.insert(1, nn.Dropout(DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        # last layer, l_xyz <4, 16, 3>, l_feature <4, 512, 16>
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, categories)
        return pred_cls


def get_feature_extractor(is_msg=True, input_channels=6, use_xyz=True, bn=True):
    if is_msg:
        model = Pointnet2MSG_Feature(
            input_channels=input_channels, 
            use_xyz=use_xyz, 
            bn=bn
        )
    else:
        model = Pointnet2SSG_Feature( 
        input_channels=input_channels, 
        use_xyz=use_xyz, 
        bn=bn
    )

    return model





def get_feature_extractor_tta(is_msg=True, input_channels=6, use_xyz=True, bn=True):
    
    model = TTA( 
    input_channels=input_channels, 
    use_xyz=use_xyz, 
    bn=bn
)
    
    return model


class Pointnet2MSG_Feature(nn.Module):
    def __init__(self, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        NPOINTS = [1024, 256, 64, 16]
        RADIUS = [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
        NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
        MLPS = [[[16, 16, 32], [32, 32, 64]],
                [[64, 64, 128], [64, 96, 128]],
                [[128, 196, 256], [128, 196, 256]],
                [[256, 256, 512], [256, 384, 512]]]

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        return l_xyz, l_features


class Pointnet2SSG_Feature(nn.Module):
    def __init__(self, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        NPOINTS = [1024, 256, 64, 16]
        # NPOINTS = [2048, 512, 128, 16] #

        RADIUS = [0.02, 0.04, 0.06, 0.08]
        NSAMPLE = [32, 32, 16, 16]
        MLPS = [[32, 32, 64], [64, 64, 128],
                [128, 128, 256], [256, 256, 512]]

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            mlps = [channel_in] + mlps
            channel_out += mlps[-1]

            self.SA_modules.append(
                PointnetSAModule(
                    npoint=NPOINTS[k],
                    radius=RADIUS[k],
                    nsample=NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=bn
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        return l_xyz, l_features


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist 


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # def forward(self, x):
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

    #     attn = (q @ k.transpose(-2, -1)) * self.scale
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x
    
    def forward(self, x, y=None):    # y as q, x as q, k, v
        if y is None:
            # Self attention
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        # Self attention + Cross attention
        B, N, C = x.shape
        L = y.shape[1]
        x = torch.cat([x, y], dim=1) 
        qkv = self.qkv(x).reshape(B, N+L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q: B, num_heads, N+L, C//num_heads

        # Cross attention
        # y query
        attn = (q[:, :, N:] @ k[:, :, :].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        y = (attn @ v[:, :, :]).transpose(1, 2).reshape(B, L, C)
        y = self.proj(y)
        y = self.proj_drop(y)

        # Self attention
        attn = (q[:, :, :N] @ k[:, :, :N].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v[:, :, :N]).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, y # , attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    # def forward(self, x, y):    # y is q
    #     x = x + self.drop_path(self.attn(self.norm1(x))) 
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x
    def forward(self, x, y=None):    # y is q
        if y is None:
            x = x + self.drop_path(self.attn(self.norm1(x))) 
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        new_x = self.norm1(x)
        new_y = self.norm1(y)
        
        new_x, new_y = self.attn(new_x, new_y)
        new_x = x + self.drop_path(new_x)
        new_y = y + self.drop_path(new_y)
        
        new_x = new_x + self.drop_path(self.mlp(self.norm2(new_x)))
        new_y = new_y + self.drop_path(self.mlp(self.norm2(new_y)))
        return new_x, new_y

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos, x_mask=None, pos_mask=None):
        if x_mask is None:
            for _, block in enumerate(self.blocks):
                x = block(x + pos)
            return x
        else:    
            for _, block in enumerate(self.blocks):
                x, x_mask = block(x + pos, x_mask + pos_mask)      
            return x, x_mask

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)
    
def get_pos_embed(embed_dim, ipt_pos):
    """
    embed_dim: output dimension for each position
    ipt_pos: [B, G, 3], where 3 is (x, y, z)
    """
    B, G, _ = ipt_pos.size()
    assert embed_dim % 6 == 0
    omega = torch.arange(embed_dim // 6).float().to(ipt_pos.device) # NOTE
    omega /= embed_dim / 6.
    # (0-31) / 32
    omega = 1. / 10000**omega  # (D/6,)
    rpe = []
    for i in range(_):
        pos_i = ipt_pos[:, :, i]    # (B, G)
        out = torch.einsum('bg, d->bgd', pos_i, omega)  # (B, G, D/6), outer product
        emb_sin = torch.sin(out) # (M, D/6)
        emb_cos = torch.cos(out) # (M, D/6)
        rpe.append(emb_sin)
        rpe.append(emb_cos)
    return torch.cat(rpe, dim=-1)

class TTA(nn.Module):
    def __init__(self, input_channels=3, use_xyz=True, bn=True):
        super().__init__()

        self.trans_dim = 384   #config.trans_dim
        self.depth = 12 # config.depth
        self.drop_path_rate = 0.1 #config.drop_path_rate
        self.cls_dim = 33 # config.cls_dim
        self.num_heads = 6 # config.num_heads

        self.group_size = 32 # config.group_size
        self.num_group = 128 # config.num_group
        self.encoder_dims = 384 # config.encoder_dims
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.language_model = ClipCustom(cfg=None,num=384)
        self.language_model.freeze()

        self.pos_embed = nn.Sequential(
            nn.Linear(self.trans_dim, self.trans_dim),
            nn.GELU(),
            nn.Linear(self.trans_dim, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        # self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def load_pretrained_weight(self, weigth_path):
        if weigth_path is not None:
            ckpt = torch.load(weigth_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):  #  把键名去掉，也就是把前缀去掉
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            # if incompatible.missing_keys:
            #     print('missing_keys', logger='Transformer')
            #     print(
            #         get_missing_parameters_message(incompatible.missing_keys),
            #         logger='Transformer'
            #     )
            # if incompatible.unexpected_keys:
            #     print('unexpected_keys', logger='Transformer')
            #     print(
            #         get_unexpected_parameters_message(incompatible.unexpected_keys),
            #         logger='Transformer'
            #     )

            print(f'[Transformer] Successful Loading the ckpt from {weigth_path}')
        else:
            print('Training from scratch!!!')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
         
        

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    # clip_data is none,  guidence 是 Task 文本

    def forward(self, pointcloud: torch.cuda.FloatTensor, pointcloud_ori: torch.cuda.FloatTensor, guidence):
        xyz, features = self._break_up_pc(pointcloud)
        B, C, N = xyz.shape

        neighborhood, center = self.group_divider(xyz)
        group_input_tokens = self.encoder(neighborhood)

        pos = self.pos_embed(get_pos_embed(self.trans_dim, center))

        x = group_input_tokens
        x = self.blocks(x, pos)
        x = self.norm(x)



        xyz_ori, features_ori = self._break_up_pc(pointcloud_ori)
        B, C, N = xyz_ori.shape

        neighborhood_ori, center_ori = self.group_divider(xyz_ori)
        group_input_tokens_ori = self.encoder(neighborhood_ori)

        pos_ori = self.pos_embed(get_pos_embed(self.trans_dim, center_ori))

        x_ori = group_input_tokens_ori
        x_ori = self.blocks(x_ori, pos_ori)
        x_ori = self.norm(x_ori)
 
 
        cond_txt, text_vector = self.language_model(guidence, None)
        # cond_txt_cls = cond_txt[torch.arange(cond_txt.shape[0]), text_vector.argmax(dim=-1)]

        # print("x shape is ", x.shape)
        # print("cond_txt shape is ", cond_txt.shape)
        cond = torch.cat([x, x_ori, cond_txt], dim=1)
        print("cond shape is ", cond.shape)
        



        return cond


def pointnet2_enc_repro(c=3, num_points=2048):
    assert (num_points == 2048)  ## cannot adapt num of points here
    model = get_feature_extractor(is_msg=False, input_channels=c-3, use_xyz=True, bn=True)
    return model
def tta_enc_repro(c=3, num_points=2048):
    assert (num_points == 2048)  ## cannot adapt num of points here
    model = get_feature_extractor_tta(is_msg=False, input_channels=c-3, use_xyz=True, bn=True)
    return model

    
    