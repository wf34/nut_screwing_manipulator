
import numpy as np

from pydrake.all import (
    BsplineTrajectory,
    MultibodyPlant,
    KinematicTrajectoryOptimization,
    PiecewisePolynomial, Solve,
    PositionConstraint
)

from nut_screwing.differential_controller import create_differential_controller_on_trajectory

IIWA_DEFAULT_POSITION = [-1.57, 0.1, 0, -1.2, 0, 1.6, 0]

def get_default_plant_position_with_inf(plant, plant_context):
    iiwa_model_instance = plant.GetModelInstanceByName('iiwa')
    indices = list(map(int, plant.GetJointIndices(model_instance=iiwa_model_instance)))
    n = plant.num_positions()
    plant_0 = np.zeros(n)
    plant_inf = np.eye(n) * 1.e-9
    for i, q in zip(indices, IIWA_DEFAULT_POSITION):
        plant_0[i] = q
        plant_inf[i, i] = 1
    # print(plant_0, '\n', plant_inf)
    return  plant.GetPositions(plant_context), plant_inf


def make_wsg_command_trajectory(time_end):
    opened = np.array([0.107]);
    closed = np.array([0.0]);

    return PiecewisePolynomial.FirstOrderHold(
        [0., time_end],
        np.hstack([[opened], [opened]]))


def PublishPositionTrajectory(trajectory,
                              root_context,
                              plant,
                              visualizer,
                              time_step=1.0 / 33.0):
    """
    Args:
        trajectory: A Trajectory instance.
    """
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)

    visualizer.StartRecording(False)

    for t in np.append(
            np.arange(trajectory.start_time(), trajectory.end_time(),
                      time_step), trajectory.end_time()):
        root_context.SetTime(t)
        plant.SetPositions(plant_context, trajectory.value(t))
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()


def create_experimental_controller(builder, plant, input_iiwa_position_port, context, X_G, v):
    for n in ['initial', 'prepick']:
        assert n in X_G

    plant_context = plant.GetMyContextFromRoot(context)

    def print_model_dofs(model_name):
        iiwa_model_instance = plant.GetModelInstanceByName(model_name)
        indices = plant.GetJointIndices(model_instance=iiwa_model_instance)
        print(model_name, indices, len(indices), 'or, in more detail:')
        for i in indices:
            ith_joint = plant.get_joint(joint_index=i)
            print('{}th joint!! parent: {} child: {}; own name: {}'.format(i, ith_joint.parent_body().name(), ith_joint.child_body().name(), ith_joint.name()))

    q0, inf0 = get_default_plant_position_with_inf(plant, plant_context)
    print_model_dofs('iiwa')
    print_model_dofs('gripper')
    print_model_dofs('nut_and_bolt')

    for name in ['num_positions', 'num_velocities', 'num_joints', 'num_frames', 'num_model_instances', 'num_bodies', 'num_actuators', 'num_actuated_dofs']:
        print('{}={}'.format(name, getattr(plant, name)()))

    num_q = plant.num_positions()
    num_c = 10
    print('num_positions: {}; num control points: {}'.format(num_q, num_c))

    print('before c-tor')
    trajopt = KinematicTrajectoryOptimization(num_q, 10)
    print('after c-tor')
    prog = trajopt.get_mutable_prog()

    q_guess = np.tile(q0.reshape((num_q, 1)), (1, trajopt.num_control_points()))
    q_guess[0,:] = np.linspace(0, -np.pi/2, trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    trajopt.AddDurationCost(1)
    trajopt.AddPathLengthCost(1)

    print('after first 2')
    #iiwa_position_limits_lower = plant.GetPositionLowerLimits()[2:9]
    #iiwa_position_limits_upper = plant.GetPositionUpperLimits()[2:9]

    #iiwa_velocity_limits_lower = plant.GetVelocityLowerLimits()[2:9]
    #iiwa_velocity_limits_lower[-1] = iiwa_velocity_limits_lower[-2] # hack

    #iiwa_velocity_limits_upper = plant.GetVelocityUpperLimits()[2:9]
    #iiwa_velocity_limits_upper[-1] = iiwa_velocity_limits_upper[-2] # hack

    #for f in [MultibodyPlant.GetPositionLowerLimits, MultibodyPlant.GetPositionUpperLimits,
    #          MultibodyPlant.GetVelocityLowerLimits, MultibodyPlant.GetVelocityUpperLimits]:
    #    rez = f(plant)
    #    print(f.__name__, rez, len(rez))

    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    print('after first 3')

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0)
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0)
    print(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)
    print('after first 4')
    trajopt.AddDurationConstraint(5., 50.)

    X_WStart = X_G['initial']
    X_WGoal = X_G['prepick']
        
    gripper_frame = plant.GetBodyByName("body").body_frame()

    # start constraint
    start_constraint = PositionConstraint(plant, plant.world_frame(),
                                          X_WStart.translation(),
                                          X_WStart.translation(),
                                          gripper_frame,
                                          [0, 0.1, 0],
                                          plant_context)
    trajopt.AddPathPositionConstraint(start_constraint, 0)
    prog.AddQuadraticErrorCost(inf0, q0,
                               trajopt.control_points()[:, 0])

    # goal constraint
    goal_constraint = PositionConstraint(plant, plant.world_frame(),
                                         X_WGoal.translation(),
                                         X_WGoal.translation(),
                                         gripper_frame,
                                         [0, 0.1, 0],
                                         plant_context)
    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    #prog.AddQuadraticErrorCost(inf0, q0,
    #                           trajopt.control_points()[:, -1])

    # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 1)

    #def PlotPath(control_points):
    #    traj = BsplineTrajectory(trajopt.basis(),
    #                             control_points.reshape((3, -1)))
    #    meshcat.SetLine('positions_path',
    #                     traj.vector_values(np.linspace(0, 1, 50)))

    #prog.AddVisualizationCallback(PlotPath,
    #                              trajopt.control_points().reshape((-1,)))
    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed, even without collisions!")
        print(result.get_solver_id().name())
        print(dir(result))
        print(result.GetInfeasibleConstraintNames(prog, 1.e-2))
        #return [None] * 3
        #result = 
    if True:
        traj_X_G = trajopt.ReconstructTrajectory(result)
        print(traj_X_G.start_time(), traj_X_G.end_time(),
              traj_X_G.rows(), traj_X_G.cols())
        PublishPositionTrajectory(traj_X_G, context, plant, v)
        exit()
        traj_X_G = traj_X_G.CopyBlock(2, 0, 6, 1) # remove a hack from here

        traj_wsg_command = make_wsg_command_trajectory(traj_X_G.end_time())
        return create_differential_controller_on_trajectory(builder, plant,
                                                            input_iiwa_position_port,
                                                            traj_X_G.MakeDerivative(),
                                                            traj_wsg_command)
