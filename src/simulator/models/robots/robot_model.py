from abc import ABC, abstractmethod

import mujoco


class RobotModelBase(ABC):
    @abstractmethod
    def assemble(self, sim_spec: mujoco.MjSpec):
        pass