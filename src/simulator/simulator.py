import json
import time

import mujoco.viewer
import cv2
import pyvirtualcam
from pydantic.dataclasses import dataclass
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


@dataclass
class SimulatorResources:
    robot_specs: RobotSpecs
    scene_parameters: SceneParameters


def load_resources() -> SimulatorResources:
    with open(robot_specs_file_path) as f:
        robot_specs = RobotSpecs(**json.load(f))

    with open(scene_parameters_file_path) as f:
        scene_parameters = SceneParameters(**json.load(f))

    return SimulatorResources(robot_specs=robot_specs, scene_parameters=scene_parameters)


def configure_camera_renderer(sim_spec: mujoco.MjSpec) -> tuple[int, int]:
    width_camera = screeninfo.get_monitors()[0].width
    height_camera = round(1080 / 1920 * width_camera)

    sim_spec.visual.global_.offwidth = width_camera
    sim_spec.visual.global_.offheight = height_camera

    return width_camera, height_camera


def configure_collisions(model: mujoco.MjModel):
    """ Déplace les collisions du robot vers le groupe 3"""
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            model.geom_bodyid[geom_id]
        )

        if body_name and body_name.startswith("robot/") and model.geom_group[geom_id] == 0:
            model.geom_group[geom_id] = 3


def build_simulator_model(simulator_resources: SimulatorResources) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData, mujoco.Renderer]:
    sim_spec = mujoco.MjSpec()

    width_camera, height_camera = configure_camera_renderer(sim_spec)

    scene = SceneModel(sim_spec, simulator_resources.scene_parameters, textures_folder_path)
    robot = Robot2026(robot_urdf_path, simulator_resources.robot_specs)

    scene.add_robot(robot)

    sim_model, sim_data = scene.compile()

    renderer = mujoco.Renderer(sim_model, height=height_camera, width=width_camera)

    return sim_spec, sim_model, sim_data, renderer


def main():
    simulator_resources = load_resources()

    sim_spec, sim_model, sim_data, renderer = build_simulator_model(simulator_resources)

    configure_collisions(sim_model)

    try:
        with pyvirtualcam.Camera(width=1920, height=1080, fps=30) as camera,  mujoco.viewer.launch_passive(sim_model, sim_data) as viewer:
            print(f"Caméra active : {camera.device}")

            start_time = time.time()

            while viewer.is_running():

                step_start = time.time()

                if start_time + 2 < time.time():
                    sim_data.ctrl[0] = -0.5
                    sim_data.ctrl[1] = 0.5
                    sim_data.ctrl[2] = 0.5
                    sim_data.ctrl[3] = -0.5

                # Simulation physique
                mujoco.mj_step(sim_model, sim_data)

                # Rendu de la caméra
                renderer.update_scene(sim_data, camera="camera")
                image = renderer.render()

                camera.send(cv2.resize(image, (1920, 1080)))
                camera.sleep_until_next_frame()

                # Affichage de l'image avec OpenCV
                cv2.imshow(
                    "Camera robot",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )
                cv2.waitKey(1)

                # Synchronisation du viewer 3D
                viewer.sync()

                # Gestion du temps de simulation
                dt = sim_model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

    finally:
        renderer.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
