
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

from differential_controller import make_gripper_trajectory, make_wsg_command_trajectory

def AddMeshcatTriad(
    meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()
):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0]
    )
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(
        path + "/x-axis", Cylinder(radius, length), Rgba(1, 0, 0, opacity)
    )

    # y-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0]
    )
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(
        path + "/y-axis", Cylinder(radius, length), Rgba(0, 1, 0, opacity)
    )

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(
        path + "/z-axis", Cylinder(radius, length), Rgba(0, 0, 1, opacity)
    )
def make_gripper_frames(X_G, X_O):
    """
    Takes a partial specification with X_G["initial"] and X_O["initial"] and X_0["goal"], and 
    returns a X_G and times with all of the pick and place frames populated.
    """

    assert 'initial' in X_G
    assert 'initial' in X_O
    assert 'goal' in X_O
    # Define (again) the gripper pose relative to the object when in grasp.
    
    #p_GgraspO = [0., 0.07, 0.0] # I want to achieve this version
    p_GgraspO = [0., 0.20, 0.07] # which of these to use depends on gravity
    
    R_GgraspO = RotationMatrix.Identity() # #RotationMatrix.Identity() #MakeZRotation(-np.pi/2.0)
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    
    X_OGgrasp = X_GgraspO.inverse()

    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.15, 0.0])

    X_G["pick"] = X_O["initial"].multiply(X_OGgrasp)
    X_G["prepick"] = X_G["pick"].multiply(X_GgraspGpregrasp)

    X_G["place"] = X_O["goal"].multiply(X_OGgrasp)
    X_G["postplace"] = X_G["place"].multiply(X_GgraspGpregrasp)
    

    # I'll interpolate a halfway orientation by converting to axis angle and halving the angle.
    X_GpickGplace = X_G["pick"].inverse().multiply(X_G["place"])


    # Now let's set the timing
    times = {"initial": 0}
      
    X_GinitialGprepick = X_G["initial"].inverse().multiply(X_G["prepick"])
    times["prepick"] = times["initial"] + 10.0 #*np.linalg.norm(X_GinitialGprepick.translation())

    # Allow some time for the gripper to close.
    times["pick_start"] = times["prepick"] + 5
    X_G["pick_start"] = X_G["pick"]
    
    times["pick_end"] = times["pick_start"] + 2.0
    X_G["pick_end"] = X_G["pick"]

    time_to_rotate = 2.+10.0*np.linalg.norm(X_GpickGplace.rotation().matrix()) 
      
    times["place_start"] = times["pick_end"] + time_to_rotate + 2.0
    X_G["place_start"] = X_G["place"]
      
    times["place_end"] = times["place_start"] + 4.0
    X_G["place_end"] = X_G["place"]

    times["postplace"] = times["place_end"] + 4.0

    return X_G, times

def AddIiwaDifferentialIK(builder, plant, frame=None):
    params = DifferentialInverseKinematicsParameters(plant.num_positions(),
                                                     plant.num_velocities())
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    params.set_end_effector_angular_speed_limit(2)
    params.set_end_effector_translational_velocity_limits([-2, -2, -2],
                                                          [2, 2, 2])
    if True:  # full iiwa
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits))
        params.set_joint_centering_gain(10 * np.eye(7))
    if frame is None:
        frame = plant.GetFrameByName("body")
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True))
    return differential_ik


class PickAndPlaceTrajectory(LeafSystem):
    def __init__(self, plant, meshcat):
        LeafSystem.__init__(self)
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self._nut_body_index = plant.GetBodyByName("nut").index()
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

    def Plan(self, context, state):
        X_G = {
            "initial":
                self.get_input_port(0).Eval(context)
                [int(self._gripper_body_index)],
        }
        X_O = {
            "initial": self.get_input_port(0).Eval(context)
              [int(self._nut_body_index)],
            #"goal": RigidTransform([0, -.6, 0])
        }
        X_OinitialOgoal = RigidTransform(RotationMatrix.MakeZRotation(-np.pi / 6))
        X_O['goal'] = X_O['initial'].multiply(X_OinitialOgoal)
        #X_GgraspO = RigidTransform(RollPitchYaw(np.pi / 2, 0, 0), [0, 0.22, 0])
        #X_OGgrasp = X_GgraspO.inverse()
        #X_G["pick"] = X_O["initial"] @ X_OGgrasp
        #X_G["place"] = X_O["goal"] @ X_OGgrasp
        X_G, times = make_gripper_frames(X_G, X_O)
        print(f"Planned {times['postplace']} second trajectory.")

        if True:  # Useful for debugging
            AddMeshcatTriad(self._meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(self._meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(self._meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(self._meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = make_gripper_trajectory(X_G, times)
        traj_wsg_command = make_wsg_command_trajectory(times)

        state.get_mutable_abstract_state(int(
            self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(
            self._traj_wsg_index)).set_value(traj_wsg_command)

    def start_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().start_time()

    def end_time(self, context):
        return context.get_abstract_state(
            int(self._traj_X_G_index)).get_value().end_time()

    def CalcGripperPose(self, context, output):
        # Evaluate the trajectory at the current time, and write it to the
        # output port.
        output.set_value(context.get_abstract_state(int(
            self._traj_X_G_index)).get_value().GetPose(context.get_time()))

    def CalcWsgPosition(self, context, output):
        # Evaluate the trajectory at the current time, and write it to the
        # output port.
        output.SetFromVector(
            context.get_abstract_state(int(
                self._traj_wsg_index)).get_value().value(context.get_time()))


def add_new_differential_controller(builder, plant, measured_iiwa_state_port, iiwa_pid_controller, meshcat):
    plan = builder.AddSystem(PickAndPlaceTrajectory(plant, meshcat))
    builder.Connect(plant.get_body_poses_output_port(),
                    plan.GetInputPort("body_poses"))

    robot = iiwa_pid_controller.get_multibody_plant_for_control()

    # Set up differential inverse kinematics.
    diff_ik = AddIiwaDifferentialIK(builder, robot)

    builder.Connect(plan.GetOutputPort("X_WG"),
                    diff_ik.get_input_port(0))
    builder.Connect(measured_iiwa_state_port,
                    diff_ik.GetInputPort("robot_state"))

    return diff_ik.get_output_port(), plan.GetOutputPort("wsg_position")
