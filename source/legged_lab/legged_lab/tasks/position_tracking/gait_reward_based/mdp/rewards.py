from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import RayCaster, ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from legged_lab.tasks.position_tracking.gait_reward_based.mdp.commands import *

def task_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the task reward based on the distance to the target position and the remaining time.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.
        Tr (float): The time window before the end of the episode to start rewarding.

    Returns:
        torch.Tensor: The computed reward tensor of shape (num_envs,).
    """
    command: PoseVelocityCommand = env.command_manager.get_term(command_name)
    robot_pos = command.robot_pos_w
    target_pos = command.target_pos_w

    distance = torch.norm(robot_pos - target_pos, dim=-1)  # (num_envs,
    
    # Calculate reward using torch.where for vectorized operation
    reward = 1.0 - 0.5 * distance
    
    return reward


def stalling_penalty(env: ManagerBasedRLEnv, command_name: str, 
                     vel_threshold: float = 0.1, distance_threshold: float = 0.25) -> torch.Tensor:
    """Compute the stalling penalty based on the robot's velocity.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        command_name (str): The name of the command to retrieve target positions.
        vel_threshold (float): The threshold for considering the robot as stalling.
        distance_threshold (float): The threshold for the distance to the target.

    Returns:
        torch.Tensor: The computed penalty tensor of shape (num_envs,).
    """
    command: PoseVelocityCommand = env.command_manager.get_term(command_name)
    speed = torch.norm(command.robot_velocity_w, dim=-1)  # (num_envs,)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)

    # Condition for when to apply the reward
    condition = (speed < vel_threshold) & (distance > distance_threshold)
    
    # Calculate reward using torch.where for vectorized operation
    reward = torch.where(condition, 1.0, 0.0)

    return reward

def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    command: PoseVelocityCommand = env.command_manager.get_term(command_name)
    distance = torch.norm(command.robot_pos_w - command.target_pos_w, dim=-1)  # (num_envs,)
    return (
        dof_error
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < command_threshold)
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < command_threshold)
    )
    
def feet_air_time1(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, command_threshold: float = 0.1
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    ang = torch.abs(env.command_manager.get_command(command_name)[:, 2]) > command_threshold
    condition = vel | ang
    # Scale with gravity projection (optional, but good for stability)
    reward *= condition
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def heading_error1(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the absolute heading error between current yaw and goal direction."""
    command: PoseVelocityCommand = env.command_manager.get_term(command_name)

    # Use the command term's wrapped heading error when available.
    if hasattr(command, "heading_command_w"):
        return torch.abs(command.heading_command_w)

    target_vec = command.target_pos_w - command.robot_pos_w
    target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
    asset: Articulation = env.scene["robot"]
    heading_err = math_utils.wrap_to_pi(target_direction - asset.data.heading_w)
    return torch.abs(heading_err)


def feet_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for high feet acceleration"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_acc = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]  # (num_envs, num_feet, 3)
    penalty = torch.norm(feet_acc, dim=-1)  # (num_envs, num_feet)
    reward = torch.sum(torch.square(penalty), dim=-1)  # (num_envs,)
    return reward

def base_height_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L1 norm.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    target_height = robot.data.default_root_state[0, 2]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_pos_w[:, 2]  # fallback to current height if sensor data is invalid
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L1 squared penalty
    reward = torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return reward
    
def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
    
def flat_orientation_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-component of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)  # (num_envs,)

def heading_error(env: ManagerBasedRLEnv, command_name: str, command_threshold: float = 0.1) -> torch.Tensor:
    """Compute the heading error between the robot's current heading and the goal heading."""
    # compute the error
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    ang_vel_cmd *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < command_threshold
    return ang_vel_cmd

def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # Use current mode durations so a continuously hoisted foot is penalized every step,
    # instead of only when contact mode switches.
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    variance = torch.var(torch.clip(current_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(current_contact_time, max=0.5), dim=1
    )
    # Keep anti-hoist active even near target (small command), preventing reward loopholes at standstill.
    death_hoist_penalty = torch.sum(torch.relu(current_air_time - 1.0), dim=1)

    return variance + death_hoist_penalty

def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    
    # Calculate height error: only penalize if foot is LOWER than target height
    # error = max(0, target_height - current_height)
    foot_z_target_error = torch.clamp(target_height - footpos_in_body_frame[:, :, 2], min=0.0) # (num_envs, num_feet)
    
    # Identify swing phase: foot has significant horizontal velocity
    is_swing = torch.norm(footvel_in_body_frame[:, :, :2], dim=2) > 0.1 # (num_envs, num_feet)

    # We sum over all feet
    reward = torch.sum(foot_z_target_error * is_swing, dim=1) # (num_envs,)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    
    # Scale with gravity projection (optional, but good for stability)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_edge_penalty(
    env: ManagerBasedRLEnv,
    FL_ray_sensor_cfg: SceneEntityCfg,
    FR_ray_sensor_cfg: SceneEntityCfg,
    RL_ray_sensor_cfg: SceneEntityCfg,
    RR_ray_sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    edge_grad_thresh: float | None = None,
    edge_curvature_thresh: float | None = None,
) -> torch.Tensor:
    """Penalize if the feet are close to the edge of the terrain.
    
    This is detected by checking gradient/curvature in the vicinity of the feet.
    """
    FL_ray_sensor: RayCaster = env.scene.sensors[FL_ray_sensor_cfg.name]
    FR_ray_sensor: RayCaster = env.scene.sensors[FR_ray_sensor_cfg.name]
    RL_ray_sensor: RayCaster = env.scene.sensors[RL_ray_sensor_cfg.name]
    RR_ray_sensor: RayCaster = env.scene.sensors[RR_ray_sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
   
    # ray_hits shape: (num_envs， 4, num_rays, 3)
    FL_ray_hits = FL_ray_sensor.data.ray_hits_w.view(env.num_envs, 1, -1, 3)
    FR_ray_hits = FR_ray_sensor.data.ray_hits_w.view(env.num_envs, 1, -1, 3)
    RL_ray_hits = RL_ray_sensor.data.ray_hits_w.view(env.num_envs, 1, -1, 3)
    RR_ray_hits = RR_ray_sensor.data.ray_hits_w.view(env.num_envs, 1, -1, 3)
    ray_hits = torch.cat([FL_ray_hits, FR_ray_hits, RL_ray_hits, RR_ray_hits], dim=1)
    
    # Get heights
    scan_heights = ray_hits[..., 2]  # (num_envs, 4, num_rays)
    valid_mask = torch.isfinite(scan_heights)
    valid_counts = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    mean_heights = torch.where(valid_mask, scan_heights, torch.zeros_like(scan_heights)).sum(dim=-1, keepdim=True)
    mean_heights = mean_heights / valid_counts
    scan_heights = torch.where(valid_mask, scan_heights, mean_heights)

    pattern_cfg = FL_ray_sensor.cfg.pattern_cfg
    if hasattr(pattern_cfg, "resolution") and hasattr(pattern_cfg, "size"):
        num_x = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        num_y = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        if getattr(pattern_cfg, "ordering", "xy") == "xy":
            grid_h, grid_w = num_y, num_x
        else:
            grid_h, grid_w = num_x, num_y
    else:
        num_rays = scan_heights.shape[-1]
        grid_h = int(num_rays ** 0.5)
        grid_w = num_rays // grid_h

    height_map = scan_heights.view(env.num_envs * 4, 1, grid_h, grid_w)

    sobel_x = height_map.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    sobel_y = height_map.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    laplace = height_map.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

    grad_x = F.conv2d(height_map, sobel_x.view(1, 1, 3, 3), padding=1)
    grad_y = F.conv2d(height_map, sobel_y.view(1, 1, 3, 3), padding=1)
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).view(env.num_envs, 4, -1)

    curv_mag = F.conv2d(height_map, laplace.view(1, 1, 3, 3), padding=1).abs().view(env.num_envs, 4, -1)

    grad_max = grad_mag.max(dim=-1)[0]
    curv_max = curv_mag.max(dim=-1)[0]
    
    # Check if foot is in contact
    # We use the robot's contact force data
    contacts = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0 # (num_envs, num_feet)
    
    # Penalty
    # If in contact AND gradient/curvature indicates edge
    is_edge = (grad_max > edge_grad_thresh) & (curv_max > edge_curvature_thresh)
    is_edge = is_edge | (~valid_mask.all(dim=-1))
    
    penalty = torch.sum(torch.where(contacts & is_edge, 1.0, 0.0), dim=-1)
    
    return penalty
