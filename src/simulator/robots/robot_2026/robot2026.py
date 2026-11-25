import random
from pathlib import Path
from typing import override

import mujoco

from simulator.robots.robot_model import RobotModelBase
from simulator.specs.robot_specs import RobotSpecs

WHEEL_HUBS = ["wheel1_hub", "wheel1_hub_2", "wheel2_hub", "wheel2_hub_2"]
ROBOT_HEIGHT = 0.218  # hauteur du châssis (m)


class Robot2026(RobotModelBase):
    def __init__(self, model_path: Path, robot_specs: RobotSpecs, rng: random.Random | None = None) -> None:
        self.model_path = model_path
        self.robot_specs = robot_specs
        # Générateur aléatoire des paramètres moteurs et frottements (injectable pour la reproductibilité)
        self.rng = rng if rng is not None else random.Random()

    @override
    def assemble(self, sim_spec: mujoco.MjSpec) -> mujoco.MjSpec:
        #TODO: masse du robot à corriger : le URDF donne 15.07 kg (châssis 12.91 kg, 4 hubs 1.99 kg, 36 rouleaux 0.16 kg), alors que robot2026.json indiquait 4 kg
        robot_spec = mujoco.MjSpec.from_file(str(self.model_path))
        self._set_shell_mesh_inertia(robot_spec)
        self._attach_to_scene(sim_spec, robot_spec)
        self._add_aruco_marker(sim_spec)
        self._convert_rollers_to_ellipsoids(sim_spec)
        self._exclude_roller_contacts(sim_spec)
        self._configure_wheels(sim_spec)
        return robot_spec

    @staticmethod
    def _set_shell_mesh_inertia(robot_spec: mujoco.MjSpec) -> None:
        """Les sous-maillages visuels sont des surfaces ouvertes : MuJoCo doit calculer leur inertie en mode shell"""
        for mesh in robot_spec.meshes:
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL

    @staticmethod
    def _attach_to_scene(sim_spec: mujoco.MjSpec, robot_spec: mujoco.MjSpec) -> None:
        """Attache le robot à la scène, avec un préfixe de noms et un joint libre sur le châssis"""
        frame = sim_spec.worldbody.add_frame(pos=[0, 0, 0.05], quat=[1, 0, 0, 0])
        sim_spec.attach(robot_spec, frame=frame, prefix="robot/")
        sim_spec.body("robot/base_link").add_freejoint()  # le châssis peut bouger sur le sol

    @staticmethod
    def _add_aruco_marker(sim_spec: mujoco.MjSpec) -> None:
        """Ajoute le tag ArUco sur le dessus du châssis"""
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

    @staticmethod
    def _convert_rollers_to_ellipsoids(sim_spec: mujoco.MjSpec) -> None:
        """Les rouleaux deviennent des ellipsoides : demi-axes x et y = rayon, demi-axe z = demi-longueur"""
        for body in sim_spec.bodies:
            if body.name.startswith("robot/") and "roller" in body.name:
                for geom in body.geoms:
                    if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        radius, half_length = geom.size[0], geom.size[1]
                        geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
                        geom.size = [radius, radius, half_length]

    @staticmethod
    def _exclude_roller_contacts(sim_spec: mujoco.MjSpec) -> None:
        """Exclut les contacts entre rouleaux voisins d'un même hub, et entre rouleaux et châssis"""
        # Rouleaux voisins d'un même hub (ils se touchent au repos)
        for hub_name in WHEEL_HUBS:
            rollers = [roller.name for roller in sim_spec.body(f"robot/{hub_name}").bodies]
            for i, roller_name in enumerate(rollers):
                neighbour_name = rollers[(i + 1) % len(rollers)]
                sim_spec.add_exclude(name=f"{roller_name}_{neighbour_name}_excl", bodyname1=roller_name, bodyname2=neighbour_name)

        # Rouleaux et châssis (le cylindre de collision du châssis les recouvre)
        for body in sim_spec.bodies:
            if body.name.startswith("robot/") and "roller" in body.name:
                sim_spec.add_exclude(name=f"{body.name}_base_excl", bodyname1="robot/base_link", bodyname2=body.name)

    def _configure_wheels(self, sim_spec: mujoco.MjSpec) -> None:
        """Configure le moteur et le frottement de chaque roue"""
        for hub_name in WHEEL_HUBS:
            hub = sim_spec.body(f"robot/{hub_name}")
            self._configure_motor(sim_spec, hub_name, hub)
            self._configure_roller_friction(hub)

    def _configure_motor(self, sim_spec: mujoco.MjSpec, hub_name: str, hub: mujoco.MjsBody) -> None:
        """Paramètre le moteur d'une roue : tirages aléatoires propres à chaque moteur"""
        motors = self.robot_specs.motors
        armature = motors.base_rotor_inertia * motors.reduction_ratio ** 2

        max_torque = self.rng.gauss(motors.locking_torque, motors.locking_torque * motors.torque_uncertainty)
        max_speed = self.rng.gauss(motors.no_load_speed, motors.no_load_speed * motors.speed_uncertainty)
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

    def _configure_roller_friction(self, hub: mujoco.MjsBody) -> None:
        """Applique un frottement de roulement aux rouleaux d'une roue : tiré une fois par roue, plancher à 0.2"""
        wheels = self.robot_specs.wheels
        roller_friction = max(0.2, self.rng.gauss(wheels.nominal_friction, wheels.nominal_friction * wheels.nominal_friction_uncertainty))
        for roller in hub.bodies:
            for geom in roller.geoms:
                geom.priority = 1
                geom.friction = [roller_friction, 0.01, 0.0001]
