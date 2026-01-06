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
from models.model.clip_sd import ClipCustom
from models.optimizer.grasp_loss_pose import GraspLossPose
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


@DIFFUSER.register()
class DexTOG(nn.Module):
    def __init__(self, eps_model: nn.Module, cfg: DictConfig, has_obser: bool, *args, **kwargs) -> None:
        super(DexTOG, self).__init__()
        
        self.eps_model = eps_model # eps model, UNetModel
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type

        self.has_observation = has_obser # used in some task giving observation

        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)
        self.other = False
        # if cfg.loss_type == 'l1':
        #     self.criterion = F.l1_loss
        # elif cfg.loss_type == 'l2':
        #     self.criterion = F.mse_loss
        # # Grasp as you say
        # elif cfg.loss_type == 'ldgd':
        #     self.criterion = GraspLossPose()
        #     self.other = True
        #             #           criterion:
        #             # hand_model:
        #             #   mjcf_path: ./data/mjcf/shadow_hand.xml
        #             #   mesh_path: ./data/mjcf/meshes
        #             #   n_surface_points: 1024
        #             #   contact_points_path: ./data/mjcf/contact_points.json
        #             #   penetration_points_path: ./data/mjcf/penetration_points.json
        #             #   fingertip_points_path: ./data/mjcf/fingertip.json
        #             # loss_weights:
        #             #   hand_chamfer: 1.0
        #             #   para: 10.0
        #             #   obj_penetration: 50.0
        #             #   self_penetration: 10.0
        #     # self.criterion = F.mse_loss
        # else:
        #     raise Exception('Unsupported loss type.')
                
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

        # ## randomly sample timesteps
        # if self.rand_t_type == 'all': 
        #     ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        # elif self.rand_t_type == 'half':
        #     ts = torch.randint(0, self.timesteps, ((B + 1) // 2, ), device=self.device)
        #     if B % 2 == 1:
        #         ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
        #     else:
        #         ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        # else:
        #     raise Exception('Unsupported rand ts type.')
        ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)  # 这个就是noisy_x = self.noise_scheduler.add_noise(X, noise, timesteps)
        x_t_copy = x_t.detach().clone()
        for i in range(x_t_copy.shape[0]):
            x_t_copy[i] = trans_denormalize(x_t_copy[i].cpu())

        x_t_hand_pose = x_t_copy[..., :48]
        x_t_hand_shape = x_t_copy[..., 48:58]
        x_t_hand_transl_obj = x_t_copy[..., 58:]
        x_t_mano_output = self.mano_layer(x_t_hand_pose.float(), x_t_hand_shape.float())
        x_t_pred_hand_pc = x_t_mano_output.verts
        x_t_hand_verts_obj = (x_t_pred_hand_pc + x_t_hand_transl_obj.unsqueeze(1)) # 生成抓取的点云

        # 得到x_t，计算x_t的点云
    

        ## predict noise

        # print("x_t shape is ", x_t.shape)
        # print("x_t type is ", x_t.dtype)
        # 这里的condtion需要计算物体点云，手部点云以及文本
        


        # condtion = self.eps_model.condition(data)   # 这里的condition()函数就是用来计算条件特征的，而且他这里的只使用了Pointnet2或是PointTransformer用于感知物理场景
        output = self.eps_model(x_t, ts, data['pos'], x_t_hand_verts_obj, data['guidence']) # DexTOG有问题，在处理原有的时候就是残缺的
        
        para_loss = torch.mean(torch.square(output - noise)) # 这个是函数的loss

        noisy_x_pred = self.q_sample(x0=data['x'], t=ts, noise=output) # noisy_x_pred = self.noise_scheduler.add_noise(X, noise_pred, timesteps)
        noisy_x_copy = noisy_x_pred.detach().clone()
        for i in range(x_t_copy.shape[0]):
            noisy_x_copy[i] = trans_denormalize(noisy_x_copy[i].cpu())

        noisy_x_hand_pose = noisy_x_copy[..., :48]
        noisy_x_hand_shape = noisy_x_copy[..., 48:58]
        noisy_x_hand_transl_obj = noisy_x_copy[..., 58:]
        noisy_x_mano_output = self.mano_layer(noisy_x_hand_pose.float(), noisy_x_hand_shape.float())
        noisy_x_pred_hand_pc = noisy_x_mano_output.verts
        noisy_x_hand_verts_obj = (noisy_x_pred_hand_pc + noisy_x_hand_transl_obj.unsqueeze(1)) # 生成抓取的点云
        # 下一步计算得到预测结果的点云，计算这两者的点云mse
        pcd_loss = F.mse_loss(x_t_hand_verts_obj, noisy_x_hand_verts_obj)

        print(f" para_loss is {para_loss}, pcd_loss is {pcd_loss}")

        # 如果把这里的condtion添加额外的参数，例如part部分，那么就可以在这里添加part的信息
        # x_t shape is torch.Size([64, 61]), ts shape is torch.Size([64]), condtion shape is torch.Size([64, 16, 512]), output shape is torch.Size([64, 61])
        loss = para_loss + pcd_loss

        return {'loss': loss, 'pcd_loss': pcd_loss, "para_loss": para_loss}
    
    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, obj_pcd, hand_pcd, guidence) -> Tuple:
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

        pred_noise = self.eps_model(x_t, t, obj_pcd, hand_pcd, guidence)
        pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise

        return pred_noise, pred_x0
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, obj_pcd, hand_pcd, guidence) -> Tuple:
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
        pred_noise, pred_x0 = self.model_predict(x_t, t, obj_pcd, hand_pcd, guidence)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
            self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, obj_pcd, hand_pcd, guidence) -> torch.Tensor:
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

        # if 'cond' in data:
        #     ## use precomputed conditional feature
        #     cond = data['cond']
        # else:
        #     ## recompute conditional feature every sampling step
        #     cond = self.eps_model.condition(data)
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, obj_pcd, hand_pcd, guidence)
        
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
        x_t_copy = x_t.detach().clone()
        for i in range(x_t_copy.shape[0]):
            x_t_copy[i] = trans_denormalize(x_t_copy[i].cpu())
        x_t_hand_pose = x_t_copy[..., :48]
        x_t_hand_shape = x_t_copy[..., 48:58]
        x_t_hand_transl_obj = x_t_copy[..., 58:]
        x_t_mano_output = self.mano_layer(x_t_hand_pose.float(), x_t_hand_shape.float())
        x_t_pred_hand_pc = x_t_mano_output.verts
        x_t_hand_verts_obj = (x_t_pred_hand_pc + x_t_hand_transl_obj.unsqueeze(1)) # 生成抓取的点云
            

        ## apply observation to x_t
        # x_t = self.apply_observation(x_t, data)
        
        ## precompute conditional feature, which will be used in every sampling step

        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data['pos'], x_t_hand_verts_obj, data['guidence']) # 这里的p_sample()函数就是单次采样过程
            ## apply observation to x_t
            # x_t = self.apply_observation(x_t, data)
            
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
    


    