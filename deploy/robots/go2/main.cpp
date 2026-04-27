#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<SportModeState_t> FSMState::sportstate = nullptr;
std::shared_ptr<HeightMap_t> FSMState::heightmap = nullptr;
std::shared_ptr<HeightMap_t> FSMState::local_target = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    FSMState::sportstate = std::make_shared<SportModeState_t>();

    auto height_map_topic = std::string("rt/heightmap");
    auto local_target_topic = std::string("rt/local_target_pos_b");
    int local_target_timeout_ms = 200;
    auto velocity_cfg = param::config["FSM"]["Velocity"];
    if (velocity_cfg && velocity_cfg["height_map_topic"]) {
        height_map_topic = velocity_cfg["height_map_topic"].as<std::string>();
    }
    if (velocity_cfg && velocity_cfg["local_target_topic"]) {
        local_target_topic = velocity_cfg["local_target_topic"].as<std::string>();
    }
    if (velocity_cfg && velocity_cfg["local_target_timeout_ms"]) {
        local_target_timeout_ms = velocity_cfg["local_target_timeout_ms"].as<int>();
    }
    FSMState::heightmap = std::make_shared<HeightMap_t>(height_map_topic);
    FSMState::local_target = std::make_shared<HeightMap_t>(local_target_topic);
    FSMState::sportstate->set_timeout_ms(200);
    FSMState::heightmap->set_timeout_ms(200);
    FSMState::local_target->set_timeout_ms(local_target_timeout_ms);

    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
    spdlog::info("Optional height-map topic: {}", height_map_topic);
    spdlog::info("Optional local-target topic: {}", local_target_topic);
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "And then press [Start] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
