from __future__ import annotations

import torch
import torch.nn.functional as F
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCaster
from isaaclab.envs.mdp import ManagerTermBase
from typing import TYPE_CHECKING
from legged_lab.tasks.position_tracking.gait_reward_based.mdp.commands import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# def remaining_time_fraction(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Returns the remaining time fraction in the episode."""
#     if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
#         env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
#     remaining_time = 1.0 - (env.episode_length_buf[:, None] * env.step_dt) / env.max_episode_length
#     return remaining_time

def target_pos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    "Target position relative to the robot's base in world frame, shape (N, 2)."
    command: PoseVelocityCommand = env.command_manager.get_term(command_name)
    return command.pose_command

class HeightScanRand(ManagerTermBase):
    
    def __init__(self, cfg: ObsTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Pre-allocate Sobel kernels for edge detection
        self.sobel_x = torch.tensor([[-1., 0., 1.],
                                     [-2., 0., 2.],
                                     [-1., 0., 1.]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[-1., -2., -1.],
                                     [ 0.,  0.,  0.],
                                     [ 1.,  2.,  1.]], dtype=torch.float32)
        self._history_buffer = None
        self._history_write_idx = 0
        self._delay_frames = None
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        grid_height: int | None = None,
        grid_width: int | None = None,
        dropout_prob: float = 0.05,
        missing_value: float = 0.0,
        enable_edge_noise: bool = True,
        edge_grad_threshold: float = 0.05,
        edge_noise_std: float = 0.03,
        random_delay_max_frames: int = 0,
        random_delay_resample_prob: float = 0.1,
    ) -> torch.Tensor:
        """对 scanner 输出做增强版随机化，返回 (N, H, W) 高度观测.

        包含：
            1. 高斯噪声
            2. 随机 dropout
            3. 腿遮挡模型
            4. 地形边缘增强噪声

        Args:
            env: 环境实例
            cfg: 随机化配置
            asset_cfg: 用于获取脚位置的资产配置
            sensor_cfg: 用于获取扫描器的传感器配置

        Returns:
            processed_scan: (N, H, W) 随机化后的高度栅格
        """
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        
        # 1. 准备数据
        # 获取世界坐标系下的击中点 (N, R, 3)
        ray_hits = sensor.data.ray_hits_w
        ray_hits_valid = torch.isfinite(ray_hits).all(dim=-1)
        ray_hits = torch.nan_to_num(ray_hits, nan=0.0, posinf=0.0, neginf=0.0)
        # 计算高度/距离: height = sensor_height - hit_point_z - offset
        # 注意：这里假设用户想要的是相对高度/距离作为 z 分量
        heights = sensor.data.pos_w[:, 2].unsqueeze(1) - ray_hits[..., 2] - 0.5
        heights = torch.where(ray_hits_valid, heights, torch.full_like(heights, missing_value))
        
        # 构造 scan_points (N, R, 3)，其中 z 分量替换为计算出的高度
        scan_points = torch.stack([ray_hits[..., 0], ray_hits[..., 1], heights], dim=-1)
        
        num_envs, num_rays, _ = scan_points.shape

        if (grid_height is None) != (grid_width is None):
            raise ValueError("grid_height and grid_width must be both set or both None.")
        if grid_height is None and grid_width is None:
            pattern_cfg = getattr(getattr(sensor, "cfg", None), "pattern_cfg", None)
            if pattern_cfg is None:
                raise ValueError(
                    "Cannot infer grid shape automatically. Please set grid_height/grid_width explicitly."
                )
            resolution = float(getattr(pattern_cfg, "resolution"))
            size_x, size_y = getattr(pattern_cfg, "size")
            grid_height = int(round(float(size_x) / resolution)) + 1
            grid_width = int(round(float(size_y) / resolution)) + 1

        H, W = grid_height, grid_width
        assert H * W == num_rays, (
            f"grid_height * grid_width must equal num_rays_per_scan, "
            f"but {H} * {W} != {num_rays}"
        )

        # 提取 xyz，并 reshape 成 (N, H, W) 以便进行 2D 处理
        xs = scan_points[..., 0].view(num_envs, H, W)
        ys = scan_points[..., 1].view(num_envs, H, W)
        zs = scan_points[..., 2].view(num_envs, H, W)

        # Clone 以避免修改原始数据
        h = zs.clone()

        # 2. 应用各种噪声
        # h = self._apply_dropout(h, dropout_prob, missing_value)
        h = self._apply_edge_noise(h, enable_edge_noise, edge_grad_threshold, edge_noise_std)

        # 3. 应用时间延迟随机化（延迟缓冲按展平形式维护）
        h_flat = h.view(num_envs, num_rays)
        h_flat = self._apply_random_delay(
            h=h_flat,
            env=env,
            max_delay_frames=random_delay_max_frames,
            resample_prob=random_delay_resample_prob,
        )

        return h_flat

    def _apply_random_delay(self, h, env, max_delay_frames, resample_prob):
        """Apply per-environment random frame delay using a rolling history buffer."""
        max_delay_frames = max(0, int(max_delay_frames))
        if max_delay_frames == 0:
            self._history_buffer = None
            self._delay_frames = None
            self._history_write_idx = 0
            return h

        num_envs, num_rays = h.shape
        buffer_len = max_delay_frames + 1

        needs_reinit = (
            self._history_buffer is None
            or self._history_buffer.device != h.device
            or self._history_buffer.shape != (buffer_len, num_envs, num_rays)
        )
        if needs_reinit:
            self._history_buffer = h.unsqueeze(0).repeat(buffer_len, 1, 1)
            self._delay_frames = torch.zeros(num_envs, dtype=torch.long, device=h.device)
            self._history_write_idx = 0

        self._history_buffer[self._history_write_idx] = h

        if self._delay_frames is None or self._delay_frames.numel() != num_envs:
            self._delay_frames = torch.zeros(num_envs, dtype=torch.long, device=h.device)

        resample_prob = float(min(max(resample_prob, 0.0), 1.0))
        if resample_prob > 0.0:
            resample_mask = torch.rand(num_envs, device=h.device) < resample_prob
            if resample_mask.any():
                self._delay_frames[resample_mask] = torch.randint(
                    low=0,
                    high=buffer_len,
                    size=(int(resample_mask.sum().item()),),
                    device=h.device,
                )

        history_available = torch.clamp(env.episode_length_buf.to(dtype=torch.long), min=0, max=max_delay_frames)
        effective_delay = torch.minimum(self._delay_frames, history_available)
        read_idx = (self._history_write_idx - effective_delay) % buffer_len

        delayed_h = self._history_buffer[read_idx, torch.arange(num_envs, device=h.device), :]
        self._history_write_idx = (self._history_write_idx + 1) % buffer_len
        return delayed_h

    def _apply_dropout(self, h, dropout_prob, missing_value):
        """应用随机 Dropout"""
        if dropout_prob > 0.0:
            drop_mask = (torch.rand_like(h) < dropout_prob)
            h = torch.where(drop_mask, torch.full_like(h, missing_value), h)
        return h

    def _apply_edge_noise(self, h, enable_edge_noise, edge_grad_threshold, edge_noise_std):
        """应用地形边缘增强噪声 (向量化实现)"""
        if not enable_edge_noise or edge_noise_std <= 0.0:
            return h
            
        # 确保 Sobel 核在正确的设备上
        if self.sobel_x.device != h.device:
            self.sobel_x = self.sobel_x.to(h.device)
            self.sobel_y = self.sobel_y.to(h.device)

        # 准备卷积输入 (N, 1, H, W)
        h_in = h.unsqueeze(1)
        
        # 计算梯度
        grad_x = F.conv2d(h_in, self.sobel_x.view(1, 1, 3, 3), padding=1)
        grad_y = F.conv2d(h_in, self.sobel_y.view(1, 1, 3, 3), padding=1)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).squeeze(1) # (N, H, W)
        
        # 应用噪声
        edge_mask = grad_mag > edge_grad_threshold
        if edge_mask.any():
            edge_noise = torch.randn_like(h) * edge_noise_std
            h = torch.where(edge_mask, h + edge_noise, h)
            
        return h