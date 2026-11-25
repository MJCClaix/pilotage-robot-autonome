import json
import time

import mujoco.viewer

from simulator.paths import (
    robot_specs_file_path,
    robot_urdf_path,
    scene_parameters_file_path,
    textures_folder_path,
)
from simulator.robots.robot_2026.robot2026 import Robot2026
from simulator.scenes.scene_model import SceneModel
from simulator.specs.robot_specs import RobotSpecs
from simulator.specs.scene_parameters import SceneParameters

if __name__ == "__main__":
    with open(robot_specs_file_path) as f:
        robot_specs = RobotSpecs(**json.load(f))

    with open(scene_parameters_file_path) as f:
        scene_parameters = SceneParameters(**json.load(f))

    sim_spec = mujoco.MjSpec()

    scene = SceneModel(sim_spec, scene_parameters, textures_folder_path)
    robot = Robot2026(robot_urdf_path, robot_specs)

    scene.add_robot(robot)

    model, data = scene.compile()

    # Déplace les collisions du robot vers le groupe 3 (masqué par défaut dans le viewer) :
    # seul le mesh visuel reste affiché. La physique n'est pas modifiée.
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id])
        if body_name and body_name.startswith("robot/") and model.geom_group[geom_id] == 0:
            model.geom_group[geom_id] = 3

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():

            step_start = time.time()

            if start_time + 2 < time.time():
                data.ctrl[0] = -0.5
                data.ctrl[1] = 0.5
                data.ctrl[2] = 0.5
                data.ctrl[3] = -0.5

            # noinspection argument-list
            mujoco.mj_step(model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)
