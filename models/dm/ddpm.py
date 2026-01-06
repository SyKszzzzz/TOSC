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

from models.optimizer.grasp_loss_pose import GraspLossPose

@DIFFUSER.register()
class DDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, cfg: DictConfig, has_obser: bool, *args, **kwargs) -> None:
        super(DDPM, self).__init__()
        
        self.eps_model = eps_model # eps model, UNetModel
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type

        self.has_observation = has_obser # used in some task giving observation

        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)
        self.other = False
        if cfg.loss_type == 'l1':
            self.criterion = F.l1_loss
        elif cfg.loss_type == 'l2':
            self.criterion = F.mse_loss
        # Grasp as you say
        elif cfg.loss_type == 'ldgd':
            self.criterion = GraspLossPose()
            self.other = True
                    #           criterion:
                    # hand_model:
                    #   mjcf_path: ./data/mjcf/shadow_hand.xml
                    #   mesh_path: ./data/mjcf/meshes
                    #   n_surface_points: 1024
                    #   contact_points_path: ./data/mjcf/contact_points.json
                    #   penetration_points_path: ./data/mjcf/penetration_points.json
                    #   fingertip_points_path: ./data/mjcf/fingertip.json
                    # loss_weights:
                    #   hand_chamfer: 1.0
                    #   para: 10.0
                    #   obj_penetration: 50.0
                    #   self_penetration: 10.0
            # self.criterion = F.mse_loss
        else:
            raise Exception('Unsupported loss type.')
                
        self.optimizer = None
        self.planner = None
        
    @property
    def device(self):
        return self.betas.device
    
    def apply_observation(self, x_t: torch.Tensor, data: Dict) -> torch.Tensor:  # False
        """ Apply observation to x_t, if self.has_observation if False, this method will return the input

        Args:
            x_t: noisy x in step t
            data: original data provided by dataloader
        """
        ## has start observation, used in path planning and start-conditioned motion generation
        if self.has_observation and 'start' in data:
            start = data['start'] # <B, T, D>
            T = start.shape[1]
            x_t[:, 0:T, :] = start[:, 0:T, :].clone()
        
            if 'obser' in data:
                obser = data['obser']
                O = obser.shape[1]
                x_t[:, T:T+O, :] = obser.clone()
        
        return x_t
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ Forward difussion process, $q(x_t \mid x_0)$, this process is determinative 
        and has no learnable parameters.

        $x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon$

        Args:
            x0: samples at step 0
            t: diffusion step
            noise: Gaussian noise
        
        Return:
            Diffused samples
        """
        B, *x_shape = x0.shape
        
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
            self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise

        return x_t

    def forward(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition

        Args:
            data: test data, data['x'] gives the target data, data['y'] gives the condition
        
        Return:
            Computed loss
        """
        # print("data['x'] type is ", type(data['x']))
        # print("data['x'] type is ", data['x'])
        data['x'] = torch.tensor(data['x'], device=self.device)
        # print("data['x'] type is ", type(data['x']))
        B = data['x'].shape[0]

        ## randomly sample timesteps
        if self.rand_t_type == 'all':
            ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        elif self.rand_t_type == 'half':
            ts = torch.randint(0, self.timesteps, ((B + 1) // 2, ), device=self.device)
            if B % 2 == 1:
                ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
            else:
                ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        else:
            raise Exception('Unsupported rand ts type.')
        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise) 
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames
        x_t = self.apply_observation(x_t, data) # equal to x_t = x_t

        ## predict noise

        # print("x_t shape is ", x_t.shape)
        # print("x_t type is ", x_t.dtype)
        condtion = self.eps_model.condition(data)   # 这里的condition()函数就是用来计算条件特征的，而且他这里的只使用了Pointnet2或是PointTransformer用于感知物理场景
        # 如果把这里的condtion添加额外的参数，例如part部分，那么就可以在这里添加part的信息
        print("x_t shape is ", x_t.shape, "ts shape is ", ts.shape , "condtion shape is ", condtion.shape) 
        # x_t shape is torch.Size([64, 61]), ts shape is torch.Size([64]), condtion shape is torch.Size([64, 16, 512]), output shape is torch.Size([64, 61])
        output = self.eps_model(x_t, ts, condtion)  

        print(f"x_t shape is {x_t.shape}, ts shape is {ts.shape}, condtion shape is {condtion.shape}, output shape is {output.shape}")
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants
        output = self.apply_observation(output, data) # equal to output = output

        ## calculate loss
        
        # "pred_pose_norm": output,
        if self.other:
            print("1111111111111111111111111111111111111111111111111111111111111111")
            B, *x_shape = x_t.shape
            pred_x0 = self.sqrt_recip_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[ts].reshape(B, *((1, ) * len(x_shape))) * output
            # 这里是最终得到x_0。 output是噪声

            
            pred_dict = {
            "noise":noise,
            "pred_pose_norm": output,
            }
            losses = self.criterion(pred_dict, data)
            return losses
            # 输入变量不一样，需要额外写
        else:
            print("22222222222222222222222222222222222222222222222222222222222222")
            loss = self.criterion(output, noise)
        # loss = self.criterion(output, noise)
        # 这个得要针对

        return {'loss': loss}
    
    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Get and process model prediction

        $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        B, *x_shape = x_t.shape

        pred_noise = self.eps_model(x_t, t, cond)
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> Tuple:
        """ Calculate the mean and variance, we adopt the following first equation.

        $\tilde{\mu} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0$
        $\tilde{\mu} = \frac{1}{\sqrt{\alpha}_t}(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t)$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        pred_noise, pred_x0 = self.model_predict(x_t, t, cond)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict) -> torch.Tensor:
        """ One step of reverse diffusion process

        $x_{t-1} = \tilde{\mu} + \sqrt{\tilde{\beta}} * z$

        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature

        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        if 'cond' in data:
            ## use precomputed conditional feature
            cond = data['cond']
        else:
            ## recompute conditional feature every sampling step
            cond = self.eps_model.condition(data)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, cond)
        
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0

        ## sampling with mean updated by optimizer and planner
        if self.optimizer is not None:
            ## openai guided diffusion uses the input x to compute gradient, see
            ## https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L436
            ## But the original formular uses the computed mean?
            gradient = self.optimizer.gradient(model_mean, data, model_variance)
            model_mean = model_mean + gradient
        if self.planner is not None:
            gradient = self.planner.gradient(model_mean, data, model_variance)
            model_mean = model_mean + gradient

        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x
    
    @torch.no_grad()
    def p_sample_loop(self, data: Dict) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling

        Args:
            data: test data, data['x'] gives the target data shape
        
        Return:
            Sampled data, <B, T, ...>
        """
        x_t = torch.randn_like(data['x'], device=self.device)
        ## apply observation to x_t
        x_t = self.apply_observation(x_t, data)
        
        ## precompute conditional feature, which will be used in every sampling step
        condition = self.eps_model.condition(data)
        data['cond'] = condition

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data) # 这里的p_sample()函数就是单次采样过程
            ## apply observation to x_t
            x_t = self.apply_observation(x_t, data)
            
            all_x_t.append(x_t)
        return torch.stack(all_x_t, dim=1)
    
    @torch.no_grad()
    def sample(self, data: Dict, k: int=1) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        In this method, the sampled results are unnormalized and converted to absolute representation.

        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ## TODO ddim sample function
        ksamples = []
        for _ in range(k):
            ksamples.append(self.p_sample_loop(data))  # 这里的p_sample_loop()函数就是一次独立且完成的采样过程
        
        ksamples = torch.stack(ksamples, dim=1)
        
        ## for sequence, normalize and convert repr
        if 'normalizer' in data and data['normalizer'] is not None:
            O = 0
            if self.has_observation and 'start' in data:
                ## the start observation frames are replace during sampling
                _, O, _ = data['start'].shape
            ksamples[..., O:, :] = data['normalizer'].unnormalize(ksamples[..., O:, :])
        if 'repr_type' in data:
            if data['repr_type'] == 'absolute':
                pass
            elif data['repr_type'] == 'relative':
                O = 1
                if self.has_observation and 'start' in data:
                    _, O, _ = data['start'].shape
                ksamples[..., O-1:, :] = torch.cumsum(ksamples[..., O-1:, :], dim=-2)
            else:
                raise Exception('Unsupported repr type.')
        
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





class ResBlock(nn.Module):

    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)
        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))
        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)
        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout
        if final_nl:
            return self.ll(Xout)

        return Xout
    
@DIFFUSER.register()
class CVAE(nn.Module):
    def __init__(self, eps_model: nn.Module,cfg: DictConfig, in_dim=61, cond_dim=1152, n_neurons=512, latentD=16, *args, **kwargs):
        super(CVAE, self).__init__()

        # Encoder layers
        self.eps_model = eps_model
        print(f"in_dim is {in_dim}, n_neurons is {n_neurons}")
        in_dim = 61
        cond_dim = 1152
        self.latentD = latentD
        self.enc_bn1 = nn.BatchNorm1d(in_dim + cond_dim)
        self.enc_rb1 = ResBlock(in_dim + cond_dim, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_dim + cond_dim, n_neurons)
        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)

        # Decoder layers
        self.dec_bn1 = nn.BatchNorm1d(cond_dim)
        self.dec_rb1 = ResBlock(latentD + cond_dim, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + cond_dim, n_neurons)
        
        self.dec_x = nn.Linear(n_neurons, in_dim)

        self.dec_pose = nn.Linear(n_neurons, 16 * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)


        if torch.cuda.is_available():
            self.device = torch.device('cuda')
    

    def mean_pooling_wo_mask(self, token_embeddings):
        # 对 token_embeddings 的第 2 维度进行求和
        sum_embeddings = torch.sum(token_embeddings, dim=1)
        # 计算 token_embeddings 的第 2 维度的长度
        length = token_embeddings.size(1)
        # 计算平均值
        return sum_embeddings / length

    def encode(self, x, condition):
        bs = x.shape[0]
        condition_pooled = self.mean_pooling_wo_mask(condition)

        print(f"x shape is {x.shape}, condition_pooled shape is {condition_pooled.shape}")
        X = torch.cat([x, condition_pooled], dim=1)
        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)
        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))

    def decode(self, Zin, condition):
        bs = Zin.shape[0]
        condition_pooled = self.mean_pooling_wo_mask(condition)
        o_condition = self.dec_bn1(condition_pooled)
        X0 = torch.cat([Zin, o_condition], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        # self.dec_x = nn.Linear(n_neurons, in_dim)
        out_x = self.dec_x(X)
    
        return out_x
    
    def forward(self, data):
        x = data['x'].to(self.device)
        condition = self.eps_model.condition(data)

        print("condition shape is ", condition.shape)
        # condition = data['y'].to(self.device)  # 假设条件信息在 data['y'] 中

        # 编码阶段
        q = self.encode(x, condition)
        z = q.rsample()  # 使用重参数化技巧从潜在分布中采样 z

        # 解码阶段
        recon_x = self.decode(z, condition)
        #  = recon_results['reconstructed_x']  # 假设这是解码器生成的重构结果

        # 计算重构损失
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')  # 或其他合适的损失函数

        # 计算 KL 散度损失
        p = torch.distributions.normal.Normal(torch.zeros_like(q.mean), torch.ones_like(q.stddev))
        kl_div = torch.distributions.kl_divergence(q, p).mean()

        # 总损失
        loss = recon_loss + kl_div

        return {'loss': loss}

    @torch.no_grad()
    def sample(self, data, seed=None, k: int=1):
        ksamples = []
        for _ in range(k):
            condition = self.eps_model.condition(data).to(self.device)  # 提取条件信息

            # 设置随机种子（如果提供）
            if seed is not None:
                torch.manual_seed(seed)

            # 从标准正态分布中采样潜在变量 Z
            bs = condition.size(0)
            Zgen = torch.randn(bs, self.latentD, device=self.device)

            # 使用解码器生成数据
            generated_results = self.decode(Zgen, condition)
            print("generated_results shape is ", generated_results.shape)
            ksamples.append(generated_results)
        ksamples = torch.stack(ksamples, dim=1)
        print(f"ksamples shape is {ksamples.shape}")
        return ksamples  # 返回生成的样本
    

    