
import numpy as np
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.geometry import (
    Cylinder,
    Rgba,
    Sphere,
)

from pydrake.all import (AbstractValue, AngleAxis, Concatenate, DiagramBuilder,
                         LeafSystem, MeshcatVisualizer, PiecewisePolynomial,
                         PiecewisePose, PointCloud, RigidTransform,
                         RollPitchYaw, Simulator, StartMeshcat,
                         DifferentialInverseKinematicsIntegrator,
                         DifferentialInverseKinematicsParameters)

from new_differential_controller import AddIiwaDifferentialIK
from run_alt_manipulator import solve_for_screwing_trajectories

class SingleTurnTrajectory(LeafSystem):
    def __init__(self, plant, meshcat, plant_context):
        LeafSystem.__init__(self)

        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))

        self.DeclareInitializationUnrestrictedUpdateEvent(self.Plan)
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose()))
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))

        self.DeclareAbstractOutputPort(
            "X_WG", lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose)
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)
        self._meshcat = meshcat
        self._plant = plant
        self._plant_context = plant_context

    def Plan(self, context, state):
        trajes = solve_for_screwing_trajectories(self._plant, self._plant_context, self._meshcat, None, None)

    def CalcGripperPose(self, context, output):
        # Evaluate the trajectory at the current time, and write it to the
        # output port.

        #output.set_value(context.get_abstract_state(int(
        #    self._traj_X_G_index)).get_value().GetPose(context.get_time()))
        pass

    def CalcWsgPosition(self, context, output):
        # Evaluate the trajectory at the current time, and write it to the
        # output port.

        #output.SetFromVector(
        #    context.get_abstract_state(int(
        #        self._traj_wsg_index)).get_value().value(context.get_time()))
        pass


def create_differential_controller(builder, plant, measured_iiwa_state_port,
                                   iiwa_pid_controller, meshcat, plant_temp_context):
    plan = builder.AddSystem(SingleTurnTrajectory(plant, meshcat, plant_temp_context))

    builder.Connect(plant.get_body_poses_output_port(), plan.GetInputPort("body_poses"))
    robot = iiwa_pid_controller.get_multibody_plant_for_control()

    diff_ik = AddIiwaDifferentialIK(builder, robot)

    builder.Connect(plan.GetOutputPort("X_WG"),
                    diff_ik.get_input_port(0))
    builder.Connect(measured_iiwa_state_port,
                    diff_ik.GetInputPort("robot_state"))

    return diff_ik.get_output_port(), plan.GetOutputPort("wsg_position")
