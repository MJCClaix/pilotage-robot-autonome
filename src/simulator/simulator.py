
import json
import time

import mujoco
import mujoco.viewer
import cv2
from screeninfo import screeninfo

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

    width_camera = screeninfo.get_monitors()[0].width
    height_camera = round(1080 / 1920 * width_camera)

    sim_spec.visual.global_.offwidth = width_camera
    sim_spec.visual.global_.offheight = height_camera

    scene = SceneModel(sim_spec, scene_parameters, textures_folder_path)
    robot = Robot2026(robot_urdf_path, robot_specs)

    scene.add_robot(robot)

    model, data = scene.compile()

    # Déplace les collisions du robot vers le groupe 3
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            model.geom_bodyid[geom_id]
        )

        if body_name and body_name.startswith("robot/") and model.geom_group[geom_id] == 0:
            model.geom_group[geom_id] = 3

    # Initialisation du renderer de la caméra
    renderer = mujoco.Renderer(model, height=height_camera, width=width_camera)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()

            while viewer.is_running():

                step_start = time.time()

                if start_time + 2 < time.time():
                    data.ctrl[0] = -0.5
                    data.ctrl[1] = 0.5
                    data.ctrl[2] = 0.5
                    data.ctrl[3] = -0.5

                # Simulation physique
                mujoco.mj_step(model, data)

                # Rendu de la caméra
                renderer.update_scene(data, camera="camera")
                image = renderer.render()

                # Affichage de l'image avec OpenCV
                cv2.imshow(
                    "Camera robot",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )
                cv2.waitKey(1)

                # Synchronisation du viewer 3D
                viewer.sync()

                # Gestion du temps de simulation
                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

    finally:
        renderer.close()
        cv2.destroyAllWindows()
