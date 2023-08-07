import functools
import itertools
import operator #methodcaller
import numpy as np
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.geometry import (
    Cylinder,
    Rgba,
    Sphere,
)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import (AbstractValue, AngleAxis, BsplineTrajectory, Concatenate, DiagramBuilder,
                         LeafSystem, MeshcatVisualizer, PiecewisePolynomial,
                         PiecewisePose, PointCloud, RigidTransform,
                         RollPitchYaw, Simulator, StartMeshcat,
                         DifferentialInverseKinematicsIntegrator,
                         DifferentialInverseKinematicsParameters)

from new_differential_controller import AddIiwaDifferentialIK
from run_alt_manipulator import solve_for_screwing_trajectories


def get_trajectory_end_time(t):
    return 1.0 if not t else t.end_time()


class SingleTurnTrajectory(LeafSystem):
    def __init__(self, plant, meshcat, plant_context):
        LeafSystem.__init__(self)

        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()]))

        self.DeclareInitializationUnrestrictedUpdateEvent(self.Plan)
        self._opt_trajectories_index = self.DeclareAbstractState(
            AbstractValue.Make([BsplineTrajectory()]))
        self._wsg_trajectory_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))

        self.DeclareAbstractOutputPort(
            "X_WG", lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose)
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)
        self._meshcat = meshcat
        self._plant = plant
        self._plant_context = plant_context


    def Plan(self, context, state):
        opt_trajectories, wsg_trajectory = \
            solve_for_screwing_trajectories(self._plant, self._plant_context, self._meshcat, None, None)
        self.total_time = functools.reduce(lambda x, y: x + get_trajectory_end_time(y), opt_trajectories, 0.)

        state.get_mutable_abstract_state(int(
            self._opt_trajectories_index)).set_value(opt_trajectories)
        state.get_mutable_abstract_state(int(
            self._wsg_trajectory_index)).set_value(wsg_trajectory)



    def get_entry(self, context):
        target_time = context.get_time()
        opt_trajectories = context.get_abstract_state(int(self._opt_trajectories_index)).get_value()
        end_times = list(itertools.accumulate(map(get_trajectory_end_time, opt_trajectories)))
        for i, (t, curr_end_time) in enumerate(zip(opt_trajectories, end_times)):
            trajectory_end_time = get_trajectory_end_time(t)
            is_last = (len(opt_trajectories) - 1) == i
            eps = 1.e-6
            assert not is_last or target_time - eps < curr_end_time, 'target_time={:.1f} end_time={:.1f} || trajes ||={} ; cur={}'.format(target_time, curr_end_time, len(opt_trajectories), i)

            if target_time > curr_end_time:
                continue
            else:
                prev_end_time = 0. if i == 0 else end_times[i-1]
                local_target_time = target_time - prev_end_time
                break

        assert 0. <= local_target_time and local_target_time <= trajectory_end_time, '{:.1f} not in ({:.1f}, {:.1f})'.format(
            target_time, 0., trajectory_end_time)

        cur_traj = opt_trajectories[i]
        if cur_traj:
            return cur_traj.value(local_target_time)
        else:
            assert 0 != i
            prev_traj = opt_trajectories[i-1]
            return prev_traj.value(prev_traj.end_time())


    def CalcGripperPose(self, context, output):
        internal_coords_at_target_time = self.get_entry(context)

        saved_internal_coords = self._plant.GetPositions(self._plant_context)
        self._plant.SetPositions(self._plant_context, internal_coords_at_target_time)
        X_WG = self._plant.EvalBodyPoseInWorld(self._plant_context, self._plant.GetBodyByName("body"))
        self._plant.SetPositions(self._plant_context, saved_internal_coords)
        output.set_value(X_WG)


    def CalcWsgPosition(self, context, output):
        target_time = context.get_time()
        wsg_trajectory = context.get_abstract_state(int(self._wsg_trajectory_index)).get_value()
        y = wsg_trajectory.value(target_time)
        output.SetFromVector(y[0])


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

    return diff_ik.get_output_port(), plan.GetOutputPort("wsg_position"), plan
