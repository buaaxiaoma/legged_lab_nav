#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

DEFAULT_RUN_DIR = Path('/home/ws/projects/legged_lab_nav/logs/rsl_rl/unitree_go2_rough/2026-04-21_12-07-51')
DEFAULT_CONFIG_DIR = Path('/home/ws/projects/legged_lab_nav/deploy/robots/go2/config')
GO2_SDK_JOINT_NAMES = [
    'FR_hip_joint',
    'FR_thigh_joint',
    'FR_calf_joint',
    'FL_hip_joint',
    'FL_thigh_joint',
    'FL_calf_joint',
    'RR_hip_joint',
    'RR_thigh_joint',
    'RR_calf_joint',
    'RL_hip_joint',
    'RL_thigh_joint',
    'RL_calf_joint',
]


def _load_yaml(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.Loader)


def _load_python_yaml(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.Loader)


def _dump_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _as_float_list(value: Any, dim: int) -> list[float]:
    if value is None:
        return [1.0] * dim
    if isinstance(value, (int, float)):
        return [float(value)] * dim
    values = [float(v) for v in value]
    if len(values) != dim:
        raise ValueError(f'expected {dim} values, got {len(values)}')
    return values


def _prod(shape: list[int]) -> int:
    return int(math.prod(shape)) if shape else 1


def _action_key(action: dict[str, Any]) -> str:
    full_path = action.get('full_path', '')
    return full_path.rsplit('.', 1)[-1] or 'JointPositionAction'


def _obs_key(obs: dict[str, Any]) -> str:
    full_path = obs.get('full_path', '')
    name = obs.get('name', '')
    if full_path.endswith('.generated_commands') or name == 'generated_commands':
        return 'velocity_commands'
    if full_path.endswith('.base_ang_vel') or name == 'base_ang_vel':
        return 'base_ang_vel'
    if full_path.endswith('.projected_gravity') or name == 'projected_gravity':
        return 'projected_gravity'
    if full_path.endswith('.joint_pos_rel') or name == 'joint_pos_rel':
        return 'joint_pos_rel'
    if full_path.endswith('.joint_vel_rel') or name == 'joint_vel_rel':
        return 'joint_vel_rel'
    if full_path.endswith('.last_action') or name == 'last_action':
        return 'last_action'
    if name == 'target_pos':
        return 'target_pos'
    if name == 'height_scan' or full_path.endswith('.HeightScanRand'):
        return 'height_scan'
    return name


def _default_params(obs_key: str, dim: int) -> dict[str, Any]:
    if obs_key in {'velocity_commands', 'keyboard_velocity_commands', 'target_pos'}:
        return {'command_name': 'base_velocity'}
    if obs_key == 'height_scan':
        return {'expected_dim': dim}
    return {}


def _map_train_to_sdk(values: list[float], train_joint_names: list[str]) -> list[float]:
    if len(values) != len(train_joint_names):
        raise ValueError('joint value count does not match training joint name count')
    sdk_values = [0.0] * len(GO2_SDK_JOINT_NAMES)
    for train_idx, joint_name in enumerate(train_joint_names):
        sdk_idx = GO2_SDK_JOINT_NAMES.index(joint_name)
        sdk_values[sdk_idx] = float(values[train_idx])
    return sdk_values


def _format_inline(values: list[float]) -> str:
    def fmt(v: float) -> str:
        text = f'{float(v):.6g}'
        return '0.0' if text == '-0' else text
    return '[' + ', '.join(fmt(v) for v in values) + ']'


def _update_fixstand_config(config_path: Path, stand_q_sdk: list[float]) -> None:
    text = config_path.read_text(encoding='utf-8')
    replacement = _format_inline(stand_q_sdk)
    pattern = re.compile(r'(\n\s*qs:\s*\[\s*\n\s*\[\],\s*\n\s*\[[^\n]*\],\s*\n\s*)\[[^\n]*\](,\s*\n\s*\])')
    new_text, count = pattern.subn(r'\1' + replacement + r'\2', text, count=1)
    if count != 1:
        raise RuntimeError(f'failed to locate FixStand.qs final waypoint in {config_path}')
    config_path.write_text(new_text, encoding='utf-8')


def _is_zero_range(range_values: Any) -> bool:
    return isinstance(range_values, (list, tuple)) and len(range_values) == 2 and abs(float(range_values[0])) < 1.0e-6 and abs(float(range_values[1])) < 1.0e-6


def _select_runtime_velocity_ranges(env_data: dict[str, Any], current_ranges: dict[str, Any]) -> dict[str, list[float]]:
    command_cfg = env_data.get('commands', {}).get('base_velocity', {}) if isinstance(env_data, dict) else {}
    velocity_ranges = command_cfg.get('velocity_ranges') or {}

    moving_ranges = []
    for terrain_name, ranges in velocity_ranges.items():
        lin_x = ranges.get('lin_vel_x')
        lin_y = ranges.get('lin_vel_y')
        ang_z = ranges.get('ang_vel_z')
        if lin_x is None or lin_y is None or ang_z is None:
            continue
        if _is_zero_range(lin_x) and _is_zero_range(lin_y) and _is_zero_range(ang_z):
            continue
        moving_ranges.append((terrain_name, ranges))

    if not moving_ranges:
        return {key: [float(v) for v in values] for key, values in current_ranges.items()}

    selected = {
        'lin_vel_x': [min(float(r['lin_vel_x'][0]) for _, r in moving_ranges), max(float(r['lin_vel_x'][1]) for _, r in moving_ranges)],
        'lin_vel_y': [min(float(r['lin_vel_y'][0]) for _, r in moving_ranges), max(float(r['lin_vel_y'][1]) for _, r in moving_ranges)],
        'ang_vel_z': [min(float(r['ang_vel_z'][0]) for _, r in moving_ranges), max(float(r['ang_vel_z'][1]) for _, r in moving_ranges)],
    }

    # Local-target control needs to be able to slow down to zero near the target.
    if selected['lin_vel_x'][0] > 0.0:
        selected['lin_vel_x'][0] = 0.0
    return selected


def _height_scan_export(env_data: dict[str, Any]) -> dict[str, Any]:
    scene = env_data.get('scene', {}) if isinstance(env_data, dict) else {}
    sensor = scene.get('height_scanner', {})
    pattern = sensor.get('pattern_cfg', {})
    policy_obs = env_data.get('observations', {}).get('policy', {}) if isinstance(env_data, dict) else {}
    height_obs = policy_obs.get('height_scan', {})
    height_params = height_obs.get('params', {})
    grid_height = int(height_params.get('grid_height', 17))
    grid_width = int(height_params.get('grid_width', 11))
    resolution = float(pattern.get('resolution', 0.1))
    size = [float(v) for v in pattern.get('size', [1.6, 1.0])]
    ordering = pattern.get('ordering', 'xy')
    offset_pos = sensor.get('offset', {}).get('pos', [0.0, 0.0, 20.0])
    ray_offset_z = float(offset_pos[2]) if len(offset_pos) >= 3 else 20.0
    expected_dim = grid_height * grid_width
    return {
        'expected_dim': expected_dim,
        'grid_height': grid_height,
        'grid_width': grid_width,
        'size': size,
        'resolution': resolution,
        'ordering': ordering,
        'ray_offset_z': ray_offset_z,
        'height_offset': 0.5,
        'frame_id': 'base_link',
    }


def _write_heightmap_config(path: Path, height_cfg: dict[str, Any]) -> None:
    _dump_yaml(path, height_cfg)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_policy_artifacts(policy_src_dir: Path, policy_output_dir: Path) -> list[tuple[Path, str]]:
    copied: list[tuple[Path, str]] = []
    policy_output_dir.mkdir(parents=True, exist_ok=True)
    missing: list[Path] = []
    for filename in ('policy.onnx', 'policy.pt'):
        src = policy_src_dir / filename
        if not src.is_file():
            missing.append(src)
            continue
        dst = policy_output_dir / filename
        shutil.copy2(src, dst)
        copied.append((dst, _sha256(dst)))
    if missing:
        missing_text = ', '.join(str(path) for path in missing)
        print(f'[WARNING] policy artifacts not found and were not copied: {missing_text}')
    return copied


def sync(
    run_dir: Path,
    deploy_yaml_path: Path,
    fsm_config_path: Path,
    heightmap_config_path: Path | None = None,
    policy_output_dir: Path | None = None,
    force_local_target_command: bool = False,
    policy_source_dir: Path | None = None,
) -> None:
    io_path = run_dir / 'io_descriptors/IO_descriptors.yaml'
    env_path = run_dir / 'params/env.yaml'
    if not io_path.is_file():
        raise FileNotFoundError(io_path)
    if not deploy_yaml_path.is_file():
        raise FileNotFoundError(deploy_yaml_path)
    if not env_path.is_file():
        raise FileNotFoundError(env_path)

    descriptors = _load_yaml(io_path)
    env_data = _load_python_yaml(env_path)
    deploy = _load_yaml(deploy_yaml_path)

    robot_desc = descriptors['articulations']['robot']
    train_joint_names = robot_desc['joint_names']
    joint_ids_map = [GO2_SDK_JOINT_NAMES.index(name) for name in train_joint_names]

    deploy['joint_ids_map'] = joint_ids_map
    scene = descriptors.get('scene', {})
    deploy['step_dt'] = float(scene.get('dt', scene.get('physics_dt', 0.005) * scene.get('decimation', 4)))
    deploy['stiffness'] = _map_train_to_sdk(robot_desc['default_joint_stiffness'], train_joint_names)
    deploy['damping'] = _map_train_to_sdk(robot_desc['default_joint_damping'], train_joint_names)
    deploy['default_joint_pos'] = [float(v) for v in robot_desc['default_joint_pos']]

    actions = {}
    for action in descriptors.get('actions', []):
        key = _action_key(action)
        actions[key] = {
            'clip': action.get('clip'),
            'joint_names': ['.*'],
            'scale': [float(v) for v in action['scale']],
            'offset': [float(v) for v in action['offset']],
            'joint_ids': None,
        }
    deploy['actions'] = actions

    policy_obs_keys = {_obs_key(obs) for obs in descriptors['observations']['policy']}
    has_target_pos_obs = 'target_pos' in policy_obs_keys

    existing_commands = deploy.setdefault('commands', {})
    if 'base_velocity' in existing_commands and 'ranges' in existing_commands['base_velocity']:
        existing_commands['base_velocity']['ranges'] = _select_runtime_velocity_ranges(
            env_data, existing_commands['base_velocity']['ranges']
        )
    if has_target_pos_obs or force_local_target_command:
        existing_commands.setdefault('base_velocity', {})['command_source'] = 'local_target'

    height_cfg = _height_scan_export(env_data)
    observations = {}
    for obs in descriptors['observations']['policy']:
        key = _obs_key(obs)
        dim = _prod(obs.get('shape', []))
        overloads = obs.get('overloads', {})
        history_length = int(overloads.get('history_length') or 1)
        params = _default_params(key, dim)
        if key == 'height_scan':
            params = height_cfg.copy()
        observations[key] = {
            'params': params,
            'clip': overloads.get('clip'),
            'scale': _as_float_list(overloads.get('scale', 1.0), dim),
            'history_length': history_length,
        }
    deploy['observations'] = observations

    _dump_yaml(deploy_yaml_path, deploy)
    if heightmap_config_path is not None:
        _write_heightmap_config(heightmap_config_path, height_cfg)

    stand_q_sdk = _map_train_to_sdk(robot_desc['default_joint_pos'], train_joint_names)
    _update_fixstand_config(fsm_config_path, stand_q_sdk)

    copied_policies: list[tuple[Path, str]] = []
    if policy_output_dir is not None:
        policy_src_dir = policy_source_dir or (run_dir / 'exported')
        copied_policies = _copy_policy_artifacts(policy_src_dir, policy_output_dir)

    print(f'[INFO] synced deploy yaml: {deploy_yaml_path}')
    print(f'[INFO] synced FixStand final q: {fsm_config_path}')
    if heightmap_config_path is not None:
        print(f'[INFO] synced MuJoCo heightmap config: {heightmap_config_path}')
    if copied_policies:
        print('[INFO] synced policy artifacts:')
        for policy_path, digest in copied_policies:
            print(f'  {policy_path} sha256={digest}')
    print(f'[INFO] deploy velocity ranges: {deploy.get("commands", {}).get("base_velocity", {}).get("ranges", {})}')
    print(f'[INFO] deploy command source: {deploy.get("commands", {}).get("base_velocity", {}).get("command_source", "keyboard")}')
    print(f'[INFO] height scan config: {height_cfg}')
    print(f'[INFO] FixStand final q (SDK order): {_format_inline(stand_q_sdk)}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Sync Go2 deploy config from a training IO_descriptors.yaml.')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--deploy-yaml', type=Path, default=DEFAULT_CONFIG_DIR / 'params/deploy.yaml')
    parser.add_argument('--fsm-config', type=Path, default=DEFAULT_CONFIG_DIR / 'config.yaml')
    parser.add_argument(
        '--heightmap-config',
        type=Path,
        default=Path('/home/ws/projects/legged_lab_nav/deploy/mujoco/simulate_python/heightmap_config.yaml'),
    )
    parser.add_argument(
        '--policy-output-dir',
        type=Path,
        default=DEFAULT_CONFIG_DIR / 'exported',
        help='Directory that receives policy.onnx/policy.pt copied from the policy source directory.',
    )
    parser.add_argument(
        '--policy-source-dir',
        type=Path,
        default=None,
        help='Directory containing policy.onnx/policy.pt. Defaults to <run-dir>/exported.',
    )
    parser.add_argument(
        '--force-local-target-command',
        action='store_true',
        default=False,
        help='Force commands.base_velocity.command_source=local_target even when target_pos is not a policy observation.',
    )
    args = parser.parse_args()
    sync(
        args.run_dir.expanduser().resolve(),
        args.deploy_yaml.expanduser().resolve(),
        args.fsm_config.expanduser().resolve(),
        args.heightmap_config.expanduser().resolve(),
        args.policy_output_dir.expanduser().resolve(),
        args.force_local_target_command,
        args.policy_source_dir.expanduser().resolve() if args.policy_source_dir is not None else None,
    )


if __name__ == '__main__':
    main()
