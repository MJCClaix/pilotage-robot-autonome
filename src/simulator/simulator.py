import json
import time
from pathlib import Path

import mujoco.viewer

from simulator.models.robots.robot_2026.robot2026 import Robot2026
from simulator.models.robots.robot_specs import RobotSpecs
from simulator.models.scenes.scene_parameters import SceneParameters

from .models.scenes.scene_2027 import SceneModel

assets_folder_path = Path("assets/simulator")
robot_specs_file_path = assets_folder_path / "components_specs/robot2026.json"
scene_parameters_file_path = assets_folder_path / "scenes_parameters/scene2026.json"
textures_folder_path = assets_folder_path / "textures"
robot_urdf_path = assets_folder_path / "URDF_models/assembled_robot2026.urdf"

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
