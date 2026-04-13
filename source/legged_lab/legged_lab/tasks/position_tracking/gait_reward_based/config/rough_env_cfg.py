from isaaclab.utils import configclass

from legged_lab.tasks.position_tracking.gait_reward_based.position_env_cfg import LocomotionPositionEnvCfg
import legged_lab.tasks.position_tracking.gait_reward_based.mdp as mdp

##
# Pre-defined configs
##
# use local assets
from legged_lab.assets.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionPositionEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # self.observations.policy.height_scan = None

        # ------------------------------Commands------------------------------
        # Smooth command changes and avoid near-zero command jitter.
        self.commands.base_velocity.resampling_time_range = (10.0, 14.0)
        self.commands.base_velocity.velocity_control_stiffness = 1.0
        self.commands.base_velocity.heading_control_stiffness = 1.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.only_positive_lin_vel_x = True
        self.commands.base_velocity.lin_vel_threshold = 0.05
        self.commands.base_velocity.ang_vel_threshold = 0.1
        self.commands.base_velocity.target_dis_threshold = 0.1
        self.commands.base_velocity.target_slowdown_distance = 0.4
        self.commands.base_velocity.enable_soft_target_slowdown = True
        self.commands.base_velocity.enable_heading_speed_gate = True
        self.commands.base_velocity.heading_speed_gate_min = 0.25
        self.commands.base_velocity.disallow_reverse_target_component = True
        self.commands.base_velocity.max_linear_cmd_step = 0.0
        self.commands.base_velocity.max_angular_cmd_step = 0.0
        self.commands.base_velocity.command_smoothing_factor = 0.05

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0), "pitch": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        #start after certain steps
        # self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200.0
        
        # Base
        self.rewards.base_lin_vel_z.weight = -2.0
        self.rewards.base_ang_vel_xy.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        
        # Task
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.stand_still.weight = -2.0
        self.rewards.stalling_penalty.weight = -1.0
        self.rewards.stalling_penalty.params["vel_threshold"] = 0.1
        self.rewards.stalling_penalty.params["distance_threshold"] = 0.3
        
        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_vel_l2.weight = -1e-4
        self.rewards.joint_acc_l2.weight = -2e-7
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.joint_vel_limits.weight = -1.0
        self.rewards.joint_deviation.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.01
        # Action penalties
        self.rewards.applied_torque_limits.weight = -0.2

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_hip", ".*_thigh", ".*_calf"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        
        # Others
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.air_time_variance.weight = -0.5
        self.rewards.air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.5
        self.rewards.feet_edge.weight = -0.1

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, "Head_.*"]

@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 32
           # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.randomize_apply_external_force_torque = None
        self.curriculum = None
