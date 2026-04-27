// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <spdlog/spdlog.h>

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

namespace
{
template <typename T>
T read_scalar_or_default(const YAML::Node& node, const char* key, T default_value)
{
    if (node && node[key]) {
        return node[key].as<T>();
    }
    return default_value;
}

float read_range_entry(const YAML::Node& ranges_node, const char* key, size_t idx, float default_value)
{
    if (!ranges_node || !ranges_node[key] || !ranges_node[key].IsSequence() || ranges_node[key].size() <= idx) {
        return default_value;
    }
    return ranges_node[key][idx].as<float>();
}

bool use_local_target_command(ManagerBasedRLEnv* env)
{
    const auto base_velocity_cfg = env->cfg["commands"]["base_velocity"];
    if (base_velocity_cfg && base_velocity_cfg["command_source"]) {
        const auto source = base_velocity_cfg["command_source"].as<std::string>();
        if (source == "local_target") {
            return true;
        }
        if (source == "joystick") {
            return false;
        }
    }
    // Auto-enable for policies that include target_pos in deploy observations.
    return env->cfg["observations"] && env->cfg["observations"]["target_pos"];
}

std::array<float, 2> read_local_target_pos_b(ManagerBasedRLEnv* env)
{
    const auto& source = env->robot->data.local_target_pos_b;
    if (source.size() >= 2) {
        return {source[0], source[1]};
    }
    return {0.0f, 0.0f};
}

std::vector<float> compute_velocity_commands_from_local_target(ManagerBasedRLEnv* env)
{
    static ManagerBasedRLEnv* cached_env = nullptr;
    static std::array<float, 3> prev_cmd = {0.0f, 0.0f, 0.0f};

    if (cached_env != env || env->episode_length == 0) {
        prev_cmd = {0.0f, 0.0f, 0.0f};
        cached_env = env;
    }

    const auto base_velocity_cfg = env->cfg["commands"]["base_velocity"];
    const auto ranges_cfg = base_velocity_cfg ? base_velocity_cfg["ranges"] : YAML::Node();

    const float lin_x_min = read_range_entry(ranges_cfg, "lin_vel_x", 0, -1.0f);
    const float lin_x_max = read_range_entry(ranges_cfg, "lin_vel_x", 1, 1.0f);
    const float lin_y_min = read_range_entry(ranges_cfg, "lin_vel_y", 0, -1.0f);
    const float lin_y_max = read_range_entry(ranges_cfg, "lin_vel_y", 1, 1.0f);
    const float ang_z_min = read_range_entry(ranges_cfg, "ang_vel_z", 0, -1.0f);
    const float ang_z_max = read_range_entry(ranges_cfg, "ang_vel_z", 1, 1.0f);

    const float velocity_k = read_scalar_or_default(base_velocity_cfg, "velocity_control_stiffness", 1.0f);
    const float heading_k = read_scalar_or_default(base_velocity_cfg, "heading_control_stiffness", 1.0f);
    const bool only_positive_lin_vel_x = read_scalar_or_default(base_velocity_cfg, "only_positive_lin_vel_x", false);
    const float lin_vel_threshold = read_scalar_or_default(base_velocity_cfg, "lin_vel_threshold", 0.0f);
    const float ang_vel_threshold = read_scalar_or_default(base_velocity_cfg, "ang_vel_threshold", 0.0f);
    const float target_dis_threshold = read_scalar_or_default(base_velocity_cfg, "target_dis_threshold", 0.1f);
    const bool enable_soft_target_slowdown = read_scalar_or_default(base_velocity_cfg, "enable_soft_target_slowdown", false);
    const float target_slowdown_distance = read_scalar_or_default(base_velocity_cfg, "target_slowdown_distance", 0.4f);
    const bool enable_heading_speed_gate = read_scalar_or_default(base_velocity_cfg, "enable_heading_speed_gate", false);
    const float heading_speed_gate_min = read_scalar_or_default(base_velocity_cfg, "heading_speed_gate_min", 0.2f);
    const bool disallow_reverse_target_component = read_scalar_or_default(base_velocity_cfg, "disallow_reverse_target_component", false);
    const float max_linear_cmd_step = read_scalar_or_default(base_velocity_cfg, "max_linear_cmd_step", 0.0f);
    const float max_angular_cmd_step = read_scalar_or_default(base_velocity_cfg, "max_angular_cmd_step", 0.0f);
    const float command_smoothing_factor = read_scalar_or_default(base_velocity_cfg, "command_smoothing_factor", 0.0f);

    const auto target_b = read_local_target_pos_b(env);
    const float target_x = target_b[0];
    const float target_y = target_b[1];
    const float target_dist = std::sqrt(target_x * target_x + target_y * target_y);
    const float heading_error = std::atan2(target_y, target_x);

    float vx = target_x * velocity_k;
    float vy = target_y * velocity_k;
    float wz = heading_error * heading_k;
    wz *= std::clamp((target_dist - 0.3f) / 0.3f, 0.0f, 1.0f);

    if (only_positive_lin_vel_x) {
        vx = std::clamp(vx, 0.0f, lin_x_max);
    } else {
        vx = std::clamp(vx, lin_x_min, lin_x_max);
    }
    vy = std::clamp(vy, lin_y_min, lin_y_max);
    wz = std::clamp(wz, ang_z_min, ang_z_max);

    if (enable_soft_target_slowdown) {
        const float slowdown_distance = std::max(target_slowdown_distance, target_dis_threshold + 1.0e-3f);
        const float stop_to_slow = slowdown_distance - target_dis_threshold;
        const float slow_scale = std::clamp((target_dist - target_dis_threshold) / (stop_to_slow + 1.0e-6f), 0.0f, 1.0f);
        vx *= slow_scale;
        vy *= slow_scale;
        wz *= slow_scale;
    }

    if (enable_heading_speed_gate) {
        float heading_scale = std::clamp(std::cos(heading_error), 0.0f, 1.0f);
        heading_scale = std::clamp(heading_scale, heading_speed_gate_min, 1.0f);
        vx *= heading_scale;
        vy *= heading_scale;
    }

    if (target_dist <= target_dis_threshold) {
        vx = 0.0f;
        vy = 0.0f;
        wz = 0.0f;
    }

    if (std::sqrt(vx * vx + vy * vy) <= lin_vel_threshold) {
        vx = 0.0f;
        vy = 0.0f;
    }
    if (std::abs(wz) <= ang_vel_threshold) {
        wz = 0.0f;
    }

    if (disallow_reverse_target_component && target_dist > target_dis_threshold) {
        const float safe_norm = std::max(target_dist, 1.0e-6f);
        const float ux = target_x / safe_norm;
        const float uy = target_y / safe_norm;
        const float forward_proj = vx * ux + vy * uy;
        if (forward_proj < 0.0f) {
            vx -= forward_proj * ux;
            vy -= forward_proj * uy;
        }
    }

    if (max_linear_cmd_step > 0.0f) {
        vx = prev_cmd[0] + std::clamp(vx - prev_cmd[0], -max_linear_cmd_step, max_linear_cmd_step);
        vy = prev_cmd[1] + std::clamp(vy - prev_cmd[1], -max_linear_cmd_step, max_linear_cmd_step);
    }
    if (max_angular_cmd_step > 0.0f) {
        wz = prev_cmd[2] + std::clamp(wz - prev_cmd[2], -max_angular_cmd_step, max_angular_cmd_step);
    }

    if (command_smoothing_factor > 0.0f) {
        const float alpha = std::clamp(command_smoothing_factor, 0.0f, 0.999f);
        vx = alpha * prev_cmd[0] + (1.0f - alpha) * vx;
        vy = alpha * prev_cmd[1] + (1.0f - alpha) * vy;
        wz = alpha * prev_cmd[2] + (1.0f - alpha) * wz;
    }

    if (only_positive_lin_vel_x) {
        vx = std::clamp(vx, 0.0f, lin_x_max);
    } else {
        vx = std::clamp(vx, lin_x_min, lin_x_max);
    }
    vy = std::clamp(vy, lin_y_min, lin_y_max);
    wz = std::clamp(wz, ang_z_min, ang_z_max);

    prev_cmd = {vx, vy, wz};
    return {vx, vy, wz};
}
} // namespace

REGISTER_OBSERVATION(base_lin_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_lin_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(actions)
{
    return last_action(env, params);
};

REGISTER_OBSERVATION(joint_vel)
{
    return joint_vel_rel(env, params);
};

REGISTER_OBSERVATION(velocity_commands)
{
    if (use_local_target_command(env)) {
        return compute_velocity_commands_from_local_target(env);
    }

    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(target_pos)
{
    if (use_local_target_command(env)) {
        const auto target_b = read_local_target_pos_b(env);
        return {target_b[0], target_b[1]};
    }
    return {0.0f, 0.0f};
}

REGISTER_OBSERVATION(height_scan)
{
    size_t expected_dim = 0;
    if (params["expected_dim"]) {
        expected_dim = params["expected_dim"].as<size_t>();
    }
    if (expected_dim == 0) {
        expected_dim = env->robot->data.height_scan.size();
    }

    const auto& source = env->robot->data.height_scan;
    if (!source.empty() && (expected_dim == 0 || source.size() == expected_dim)) {
        return source;
    }

    static bool warned_once = false;
    if (!warned_once) {
        if (source.empty()) {
            spdlog::warn("Height scan data is unavailable; using zero fallback.");
        } else {
            spdlog::warn("Height scan size mismatch (got {}, expected {}); using zero fallback.", source.size(), expected_dim);
        }
        warned_once = true;
    }

    return std::vector<float>(expected_dim, 0.0f);
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

}
}
