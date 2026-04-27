import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()
unitree_bridge = None

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)


def ConfigureViewerCamera():
    if not getattr(config, "ENABLE_TRACKING_CAMERA", True):
        return
    body_name = getattr(config, "TRACK_CAMERA_BODY", "base_link")
    try:
        track_body_id = mj_model.body(body_name).id
    except KeyError:
        print(f"Tracking camera body not found: {body_name}")
        return

    try:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    except AttributeError:
        viewer.cam.type = 1
    viewer.cam.trackbodyid = track_body_id
    viewer.cam.distance = float(getattr(config, "TRACK_CAMERA_DISTANCE", 3.0))
    viewer.cam.azimuth = float(getattr(config, "TRACK_CAMERA_AZIMUTH", 145.0))
    viewer.cam.elevation = float(getattr(config, "TRACK_CAMERA_ELEVATION", -18.0))


ConfigureViewerCamera()

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model, unitree_bridge

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    bridge_cls = UnitreeSdk2Bridge
    unitree = bridge_cls(mj_model, mj_data, locker)
    unitree_bridge = unitree

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    global unitree_bridge

    while viewer.is_running():
        locker.acquire()
        if unitree_bridge is not None:
            unitree_bridge.RenderDebugViewer(viewer)
        else:
            viewer.user_scn.ngeom = 0
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
