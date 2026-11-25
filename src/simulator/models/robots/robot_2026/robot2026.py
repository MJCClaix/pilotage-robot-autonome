import random
from pathlib import Path
from typing import override

import mujoco

from simulator.models.robots.robot_model import RobotModelBase
from simulator.models.robots.robot_specs import RobotSpecs

WHEEL_HUBS = ["wheel1_hub", "wheel1_hub_2", "wheel2_hub", "wheel2_hub_2"]
ROBOT_HEIGHT = 0.218  # hauteur du châssis (m)


class Robot2026(RobotModelBase):
    def __init__(self, model_path: Path, robot_specs: RobotSpecs):
        self.model_path = model_path
        self.robot_specs = robot_specs
    
    @override
    def assemble(self, sim_spec: mujoco.MjSpec):  # sourcery skip: low-code-quality
        #TODO: masse du robot à corriger : le URDF donne 15.07 kg (châssis 12.91 kg, 4 hubs 1.99 kg, 36 rouleaux 0.16 kg), alors que robot2026.json indiquait 4 kg
        robot_spec = mujoco.MjSpec.from_file(str(self.model_path))
        # Les sous-maillages visuels sont des surfaces ouvertes : MuJoCo doit calculer leur inertie en mode shell
        for mesh in robot_spec.meshes:
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL

        frame = sim_spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
        sim_spec.attach(robot_spec, frame=frame, prefix="robot/")
        sim_spec.body("robot/base_link").add_freejoint()  # le châssis peut bouger sur le sol

        # Tag ArUco sur le dessus du châssis
        sim_spec.body("robot/base_link").add_geom(
            name="robot/aruco_marker_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.035, 0.035, 0.00001],
            pos=[0, 0, ROBOT_HEIGHT],
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

        motors = self.robot_specs.motors
        wheels = self.robot_specs.wheels
        armature = motors.base_rotor_inertia * motors.reduction_ratio ** 2

        for hub_name in WHEEL_HUBS:
            hub = sim_spec.body(f"robot/{hub_name}")

            # Paramètres moteur : tirés une fois par roue (chaque moteur a ses propres valeurs)
            max_torque = random.gauss(motors.locking_torque, motors.locking_torque * motors.torque_uncertainty)
            max_speed = random.gauss(motors.no_load_speed, motors.no_load_speed * motors.speed_uncertainty)
            damping = max_torque / max_speed
            frictionloss = max_torque * motors.dead_zone

            wheel_joint = hub.first_joint()  # joint de roue déjà défini dans le URDF
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

            # Frottement de roulement des rouleaux de cette roue : tiré une fois par roue, plancher à 0.2
            roller_friction = max(0.2, random.gauss(wheels.nominal_friction, wheels.nominal_friction * wheels.nominal_friction_uncertainty))
            for roller in hub.bodies:
                for geom in roller.geoms:
                    geom.friction = [roller_friction, 0.01, 0.0001]

        return robot_spec
