// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "Types.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "optional_topic_logger.h"

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(
        LowStatePtr lowstate_,
        std::shared_ptr<SportModeState_t> sportstate_ = nullptr,
        std::shared_ptr<HeightMap_t> heightmap_ = nullptr,
        std::shared_ptr<HeightMap_t> local_target_ = nullptr,
        std::string sportstate_topic = "rt/sportmodestate",
        std::string heightmap_topic = "rt/heightmap",
        std::string local_target_topic = "rt/local_target_pos_b")
    : lowstate(lowstate_),
      sportstate(std::move(sportstate_)),
      heightmap(std::move(heightmap_)),
      local_target(std::move(local_target_)),
      sportstate_logger(std::move(sportstate_topic)),
      heightmap_logger(std::move(heightmap_topic)),
      local_target_logger(std::move(local_target_topic))
    {
        data.joystick = &lowstate->joystick;
        data.local_target_pos_b = {0.0f, 0.0f};
    }

    void update() override
    {
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            // base_angular_velocity
            for(int i(0); i<3; i++) {
                data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
            }
            // project_gravity_body
            data.root_quat_w = Eigen::Quaternionf(
                lowstate->msg_.imu_state().quaternion()[0],
                lowstate->msg_.imu_state().quaternion()[1],
                lowstate->msg_.imu_state().quaternion()[2],
                lowstate->msg_.imu_state().quaternion()[3]
            );
            data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
            // joint positions and velocities
            for(int i(0); i < data.joint_ids_map.size(); i++) {
                data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
                data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
            }
        }

        data.root_lin_vel_b.setZero();
        sportstate_logger.poll(sportstate);
        if (sportstate && !sportstate->isTimeout()) {
            std::lock_guard<std::mutex> lock(sportstate->mutex_);
            for (int i = 0; i < 3; ++i) {
                data.root_lin_vel_b[i] = sportstate->msg_.velocity()[i];
            }
        }

        data.height_scan.clear();
        heightmap_logger.poll(heightmap);
        if (heightmap && !heightmap->isTimeout()) {
            std::lock_guard<std::mutex> lock(heightmap->mutex_);
            const auto& map_data = heightmap->msg_.data();
            data.height_scan.assign(map_data.begin(), map_data.end());
        }

        data.local_target_pos_b = {0.0f, 0.0f};
        local_target_logger.poll(local_target);
        if (local_target && !local_target->isTimeout()) {
            std::lock_guard<std::mutex> lock(local_target->mutex_);
            const auto& target_data = local_target->msg_.data();
            if (target_data.size() >= 2) {
                data.local_target_pos_b[0] = target_data[0];
                data.local_target_pos_b[1] = target_data[1];
            }
        }
    }

    LowStatePtr lowstate;
    std::shared_ptr<SportModeState_t> sportstate;
    std::shared_ptr<HeightMap_t> heightmap;
    std::shared_ptr<HeightMap_t> local_target;
    OptionalTopicReceiptLogger sportstate_logger;
    OptionalTopicReceiptLogger heightmap_logger;
    OptionalTopicReceiptLogger local_target_logger;
};

}
