
import numpy as np

from pydrake.all import (
    BasicVector,
    LeafSystem,
    Integrator, JacobianWrtVariable,
    PiecewisePose, PiecewisePolynomial, TrajectorySource
)

# any controller basically needs to put forward its output ports for iiwa_position and wsg_position 
# controller may also depend on the measured positions

def make_gripper_trajectory(X_G, times):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    """

    sample_times = []
    poses = []
    for name in ["initial",
                 "prepick", "pick_start", "pick_end",
                 "place_start", "place_end", "postplace"]:
        sample_times.append(times[name])
        poses.append(X_G[name])
    #MakeLinear
    return PiecewisePose.MakeCubicLinearWithEndLinearVelocity(sample_times, poses)


def make_wsg_command_trajectory(times):
    opened = np.array([0.107]);
    closed = np.array([0.0]);

    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["pick_start"]], np.hstack([[opened], [opened]]))
    traj_wsg_command.AppendFirstOrderSegment(times["pick_end"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_start"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_end"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["postplace"], opened)
    return traj_wsg_command

# We can write a new System by deriving from the LeafSystem class.
# There is a little bit of boiler plate, but hopefully this example makes sense.
class PseudoInverseController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa7")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", BasicVector(6))
        self.q_port = self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.DeclareVectorOutputPort("iiwa_velocity", BasicVector(7), 
                                     self.CalcOutput)
        # TODO(russt): Add missing binding
        #joint_indices = plant.GetJointIndices(self._iiwa)
        #self.position_start = plant.get_joint(joint_indices[0]).position_start()
        #self.position_end = plant.get_joint(joint_indices[-1]).position_start()
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context, output):
        #if np.abs(context.get_time() - int(context.get_time())) < 1e-9:
        #    print('At', context.get_time())
        #    print('robot is at', plant.EvalBodyPoseInWorld(context, plant.GetBodyByName("body")))
        
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV, 
            self._G, [0,0,0], self._W, self._W)
        J_G = J_G[:,self.iiwa_start:self.iiwa_end+1] # Only iiwa terms.
        v = np.linalg.pinv(J_G).dot(V_G)
        output.SetFromVector(v)


def create_differential_controller(builder, plant, input_iiwa_position_port, X_G, times):
    # input_iiwa_position_port: measured iiwa position 
    # X_G: a dict with Gripper poses; keys - str labels of action
    # times: a dict with action times; keys - str labels of action

    # Make the trajectories
    traj = make_gripper_trajectory(X_G, times)
    print('start', traj.GetPose(time=0.).GetAsMatrix4()[2, 3])
    print('pick_start', traj.GetPose(time=10.).GetAsMatrix4()[2, 3])
    print('pick_end', traj.GetPose(time=17.).GetAsMatrix4()[2, 3])
    traj_wsg_command = make_wsg_command_trajectory(times)
    print('traj, number of segments', traj.get_number_of_segments())
    traj_V_G = traj.MakeDerivative()

    return create_differential_controller_on_trajectory(builder, plant, input_iiwa_position_port,
                                                        traj.MakeDerivative(),
                                                        traj_wsg_command)


def create_differential_controller_on_trajectory(builder,
                                                 plant,
                                                 input_iiwa_position_port,
                                                 gripper_velocity_trajectory,
                                                 wsg_grajectory):

    wsg_source = builder.AddSystem(TrajectorySource(wsg_grajectory))
    wsg_source.set_name("wsg_command")
    
    V_G_source = builder.AddSystem(TrajectorySource(gripper_velocity_trajectory))
    V_G_source.set_name("v_WG")

    controller = builder.AddSystem(PseudoInverseController(plant))
    controller.set_name("PseudoInverseController")
    builder.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))

    integrator = builder.AddSystem(Integrator(7))
    integrator.set_name("integrator")
    
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    
    builder.Connect(input_iiwa_position_port, controller.GetInputPort("iiwa_position"))
    
    return integrator.get_output_port(), wsg_source.get_output_port(), integrator

