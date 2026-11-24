import random

import mujoco

from simulator.models.robots.robot_model import RobotModelBase
from simulator.models.robots.robot_specs import RobotSpecs

WHEEL_HUBS = ["wheel1_hub", "wheel1_hub_2", "wheel2_hub", "wheel2_hub_2"]


class Robot2026(RobotModelBase):
    def __init__(self, model_path: str, robot_specs: RobotSpecs):
        self.model_path = model_path
        self.robot_specs = robot_specs

    def assemble(self, sim_spec: mujoco.MjSpec, start_position: tuple[float, float, float]):
        robot_spec = mujoco.MjSpec.from_file(self.model_path)
        # Les sous-maillages visuels sont des surfaces ouvertes : MuJoCo doit calculer leur inertie en mode shell
        for mesh in robot_spec.meshes:
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL

        frame = sim_spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
        sim_spec.attach(robot_spec, frame=frame, prefix="robot/")
        sim_spec.body("robot/base_link").add_freejoint()  # le châssis peut bouger sur le sol

        # Tag ArUco sur le dessus du châssis (z = 0.218 m, hauteur du châssis)
        sim_spec.body("robot/base_link").add_geom(
            name="robot/aruco_marker_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.035, 0.035, 0.00001],
            pos=[0, 0, 0.218],
            material="aruco_marker_material",
            group=1,  # visible (groupe des visuels)
            contype=0,
            conaffinity=0,
        )

        # Les rouleaux deviennent des ellipsoides : demi-axes x et y = rayon, demi-axe z = demi-longueur
        for body in sim_spec.bodies:
            if body.name.startswith("robot/") and "roller" in body.name:
                for geom in body.geoms:
                    if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        radius, half_length = geom.size[0], geom.size[1]
                        geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
                        geom.size = [radius, radius, half_length]

        # Exclut le contact entre rouleaux voisins d'un même hub (ils se touchent au repos)
        for hub_name in WHEEL_HUBS:
            rollers = [roller.name for roller in sim_spec.body(f"robot/{hub_name}").bodies]
            for i, roller_name in enumerate(rollers):
                neighbour_name = rollers[(i + 1) % len(rollers)]
                sim_spec.add_exclude(name=f"{roller_name}_{neighbour_name}_excl", bodyname1=roller_name, bodyname2=neighbour_name)

        # Exclut le contact entre les rouleaux et le châssis (le cylindre de collision du châssis les recouvre)
        for body in sim_spec.bodies:
            if body.name.startswith("robot/") and "roller" in body.name:
                sim_spec.add_exclude(name=f"{body.name}_base_excl", bodyname1="robot/base_link", bodyname2=body.name)

        # Paramètres moteur : tirés une fois, partagés par les 4 roues
        motors = self.robot_specs.motors
        max_torque = random.gauss(motors.locking_torque, motors.locking_torque * motors.torque_uncertainty)
        max_speed = random.gauss(motors.no_load_speed, motors.no_load_speed * motors.speed_uncertainty)
        damping = max_torque / max_speed
        armature = motors.base_rotor_inertia * motors.reduction_ratio ** 2
        frictionloss = max_torque * motors.dead_zone

        for hub_name in WHEEL_HUBS:
            wheel_joint = sim_spec.body(f"robot/{hub_name}").first_joint()  # joint de roue déjà défini dans le URDF
            wheel_joint.damping = [damping, 0, 0]
            wheel_joint.armature = armature
            wheel_joint.frictionloss = frictionloss
            sim_spec.add_actuator(
                name=f"robot/{hub_name}_cmd",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=wheel_joint.name,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                dyntype=mujoco.mjtDyn.mjDYN_NONE,
                ctrlrange=[-1.0, 1.0],
                gear=(max_torque, 0, 0, 0, 0, 0),
            )

        return robot_spec
