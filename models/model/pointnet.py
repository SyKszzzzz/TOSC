import os
import torch
import torch.nn as nn

from loguru import logger

import torch, torch.nn as nn, numpy as np, torch.nn.functional as F
from torch.autograd import Variable


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)  # in-channel, out-channel, kernel size
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
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
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]  # global descriptors

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(3).flatten().astype(np.float32))).view(1, 9).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
    
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
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
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(
            1, self.k ** 2).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False,
                 channel=3, detailed=False):
        # when input include normals, it
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)  # Batch * 3 * 3
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.detailed = detailed

    def forward(self, x):
        x = x.transpose(2, 1)
        _, D, N = x.size()  # Batch Size, Dimension of Point Features, Num of Points
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            # pdb.set_trace()
            x, feature = x.split([3, D-3], dim=2)
        x = torch.bmm(x, trans)
        # feature = torch.bmm(feature, trans)  # feature -> normals

        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        out1 = self.bn1(self.conv1(x))
        x = F.relu(out1)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        out2 = self.bn2(self.conv2(x))
        x = F.relu(out2)

        out3 = self.bn3(self.conv3(x))
        # x = self.bn3(self.conv3(x))
        x = torch.max(out3, 2, keepdim=False)[0]
        if self.global_feat:
            return x, trans, trans_feat
        elif self.detailed:
            return out1, out2, out3, x
        else:  # concatenate global and local feature together
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        


class Pointnet_SV(nn.Module):
    def __init__(self, input_channels=3, use_xyz=True, bn=True):
        super(Pointnet_SV, self).__init__()

        self.feat = PointNetEncoder(global_feat=True,
                                    feature_transform=True,
                                    channel=input_channels)
        channel_in = input_channels

        self.ins_preprocess = nn.Linear(768, 128)
        self.task_preprocess = nn.Linear(768, 128)
        self.obj_preprocess = nn.Linear(768, 128)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)



        # self.attention_part = torch.nn.MultiheadAttention(embed_dim=128, num_heads=8)
        # # self.attention_ins = MultiHeadAttention(d_model=128, n_head=8)
        # self.ffn = nn.Sequential(
        #     nn.Linear(128, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        # )
        # self.ln1 = nn.LayerNorm(128)
        # self.ln2 = nn.LayerNorm(128)
        # # self.ln1 = nn.LayerNorm()
        # self.fusion_mlp = torch.nn.Sequential(
        #     torch.nn.Linear(2*512, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 512)
        # )

        


    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask
    
    def mean_pooling_wo_mask(self, token_embeddings):
        # 对 token_embeddings 的第 1 维度进行求和
        sum_embeddings = torch.sum(token_embeddings, dim=1)
        # 计算 token_embeddings 的第 1 维度的长度
        length = token_embeddings.size(1)
        # 计算平均值
        return sum_embeddings / length

    

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features


    def mask(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        out_embeddings = token_embeddings * input_mask_expanded
        # sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return out_embeddings
    
    def load_pretrained_weight(self, weigth_path: str) -> None:
        if weigth_path is None:
            logger.info(f'Train Scene model(PointNet) from scratch...')
            return

        if not os.path.exists(weigth_path):
            raise Exception('Can\'t find pretrained point-transformer weights.')

        checkpoint = torch.load(weigth_path)
        model_dict = checkpoint['model_state_dict']


        static_dict = {}
        current_model_dict = self.state_dict()
        print("current_model_dict is ", current_model_dict.keys())
        # for key in model_dict.keys():
        #     if 'enc' in key:
        #         static_dict[key] = model_dict[key]
        static_dict = {}
        for key, value in model_dict.items():
            new_key = key.replace('module.', '')
            if new_key in current_model_dict and current_model_dict[new_key].size() == value.size():
                static_dict[new_key] = value  # 只加载匹配的参数
                logger.info(f"Logging parameter {key}.")
            else:
                logger.info(f"Skipping parameter {key}, shape mismatch or not found in the current model.")
        current_model_dict.update(static_dict)
        self.load_state_dict(current_model_dict)
        # self.load_state_dict(static_dict)
        logger.info(f'Load pretrained scene model(PointNet): {weigth_path}')

        # exit()
        

    def forward(self, pointcloud: torch.cuda.FloatTensor, ins_disc, ins_mask_disc, obj_disc, obj_mask_disc):
        # xyz, features = self._break_up_pc(pointcloud)

        # l_xyz, l_features = [xyz], [features]
        # for i in range(len(self.SA_modules)):
        #     li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
        #     l_xyz.append(li_xyz)
        #     l_features.append(li_features)

        # final_features = l_features[-1]  # (B, 512, 16)  F_G
        # B, C, N = final_features.shape   #

        batch_size, _, num_points = pointcloud.size()
 
        x, trans, trans_feat = self.feat(pointcloud) #
        print("x  shape is ", x.shape) # [B, 1088, 1024] # 这个是加上了local的
                                                                   # [B, 1024] # 这个是global的
        x = F.relu(self.bn1(self.fc1(x)))

        point_embedding = F.relu(self.bn2(self.fc2(x)))
        print("point_embedding  shape is ", point_embedding.shape) # [B, 256]

        expanded_features = point_embedding.unsqueeze(2).repeat(1, 1, 128)
        


        print(f"ins_disc shape is {ins_disc.shape}, ")
        print(f"obj_disc type {type(obj_disc)} shape is ")

        ins_embed = self.ins_preprocess(ins_disc)  # [B, 50, 128]
        obj_embed = self.obj_preprocess(obj_disc)  # [B, 150, 128]

        

        ins_embedding = self.mask(ins_embed, ins_mask_disc)      # F_I
        obj_embedding = self.mask(obj_embed, obj_mask_disc)   # F_T
        print("ins_embedding shape is ", ins_embedding.shape, "obj_embedding  shape is ", obj_embedding.shape)

     


        # L0_final_embedding
        enriched_features = torch.cat([expanded_features, ins_embedding, obj_embedding], dim=1) # # (B, 256+50+150, 128)



        return enriched_features



class PointNetEnc(nn.Module):
    def __init__(self,
                 layers_size=[64, 128, 512], c=3, num_points=2048, num_tokens=8):
        super(PointNetEnc, self).__init__()
        self.num_groups = num_tokens
        self.c = c
        self.num_points = num_points
        self.layers_size = [c] + layers_size
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activate_func = nn.ReLU()
        for i in range(len(self.layers_size) - 1):
            self.conv_layers.append(nn.Conv1d(self.layers_size[i], self.layers_size[i + 1], 1))
            self.bn_layers.append(nn.BatchNorm1d(self.layers_size[i+1]))
            nn.init.xavier_normal_(self.conv_layers[-1].weight)

        self.feat_size = layers_size[-1]
        self.togen_layers_size = [self.feat_size, 4 * self.feat_size, num_tokens * self.feat_size]
        self.togen_conv_layers = nn.ModuleList()
        self.togen_bn_layers = nn.ModuleList()
        self.togen_activate_func = nn.ReLU()
        for i in range(len(self.togen_layers_size) - 1):
            self.togen_conv_layers.append(nn.Conv1d(self.togen_layers_size[i], self.togen_layers_size[i+1], 1))
            self.togen_bn_layers.append(nn.BatchNorm1d(self.togen_layers_size[i+1]))
            nn.init.xavier_normal_(self.togen_conv_layers[-1].weight)

    def forward(self, x):
        # input: B * N * c
        # output: B * latent_size
        x = x.transpose(1, 2)
        for i in range(len(self.conv_layers) - 1):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.activate_func(x)
        x = self.bn_layers[-1](self.conv_layers[-1](x))
        x = torch.max(x, 2, keepdim=True)[0]  # B x self.feat_size x 1

        # x = x.view(-1, 1, self.layers_size[-1])
        # forward token padding(togen) layer
        for i in range(len(self.togen_conv_layers) - 1):
            print("x shape is ", x.shape)
            print("self.togen_bn_layers[i] is ", self.togen_conv_layers[i])
            x = self.togen_conv_layers[i](x)
            print("x shape is ", x.shape)
            print("self.togen_bn_layers[i] is ", self.togen_bn_layers[i])
            x = self.togen_bn_layers[i](x)
            x = self.togen_activate_func(x)
        x = self.togen_bn_layers[-1](self.togen_conv_layers[-1](x)).squeeze(2)

        return x

    def load_pretrained_weight(self, weigth_path: str) -> None:
        if weigth_path is None:
            logger.info(f'Train Scene model(PointNet) from scratch...')
            return

        if not os.path.exists(weigth_path):
            raise Exception('Can\'t find pretrained point-transformer weights.')

        model_dict = torch.load(weigth_path)
        static_dict = {}
        for key in model_dict.keys():
            if 'enc' in key:
                static_dict[key] = model_dict[key]

        self.load_state_dict(static_dict)
        logger.info(f'Load pretrained scene model(PointNet): {weigth_path}')








def pointnet_enc_repro(**kwargs):
    model = PointNetEnc([64, 128, 512], **kwargs)
    return model

def pointnet_sv_enc_repro(**kwargs):
    model = Pointnet_SV()
    return model


if __name__ == '__main__':
    random_model = pointnet_enc_repro(c=3, num_points=2048)
    dummy_inputs = torch.randn(1, 2048, 3)
    print("random_model is ", random_model)
    o = random_model(dummy_inputs)
    print()

