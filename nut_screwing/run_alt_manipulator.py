import time
import os
import sys

import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, BsplineTrajectory,
                         DiagramBuilder, KinematicTrajectoryOptimization,
                         MeshcatVisualizer, MeshcatVisualizerParams,
                         MinimumDistanceConstraint, Parser, PositionConstraint,
                         Rgba, RigidTransform, Role, Solve, Sphere,
                         StartMeshcat, FindResourceOrThrow, RevoluteJoint, RollPitchYaw, GetDrakePath, MeshcatCone)

#from scenarios import AddWsg, AddIiwa

def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# TODO: take argument for whether we want the welded fingers version or not
def AddWsg(plant,
           iiwa_model_instance,
           roll=np.pi / 2.0,
           welded=False,
           sphere=False):
    parser = Parser(plant)
    if welded:
        if sphere:
            gripper = parser.AddModelFromFile(
                FindResource("models/schunk_wsg_50_welded_fingers_sphere.sdf"),
                "gripper")
        else:
            gripper = parser.AddModelFromFile(
                FindResource("models/schunk_wsg_50_welded_fingers.sdf"),
                "gripper")
    else:
        gripper = parser.AddModelFromFile(
            FindResourceOrThrow(
                "drake/manipulation/models/"
                "wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"))

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0, 0, 0.09])
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa_model_instance),
                     plant.GetFrameByName("body", gripper), X_7G)
    return gripper


def AddIiwa(plant, collision_model="no_collision"):
    sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        f"iiwa7_{collision_model}.sdf")

    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(sdf_path)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa

def AddPlanarIiwa(plant):
    urdf = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/urdf/"
        "planar_iiwa14_spheres_dense_elbow_collision.urdf")

    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(urdf)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    q0 = [0.1, -1.2, 1.6]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa

def PublishPositionTrajectory(trajectory,
                              root_context,
                              plant,
                              visualizer,
                              meshcat=None,
                              time_step=1.0 / 33.0):
    """
    Args:
        trajectory: A Trajectory instance.
    """
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)

    visualizer.StartRecording(False)
    blue = np.array([0., 0., 1., 1.])
    green = np.array([0., 1., 0., 1.])

    dur = trajectory.end_time()  - trajectory.start_time()
    for t in np.append(
            np.arange(trajectory.start_time(), trajectory.end_time(),
                      time_step), trajectory.end_time()):
        plant.SetPositions(plant_context, trajectory.value(t))
        if meshcat:
            a = t / dur
            assert 0. <= a and a <= 1.
            b = 1. - a
            mixture = a * blue + b * green
            root_context.SetTime(t)
            X_W_G = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))
            curr_point = 'point_{}'.format(t)
            meshcat.SetObject(curr_point, Sphere(0.01), rgba=Rgba(*mixture.tolist()))
            meshcat.SetTransform(curr_point, X_W_G)

        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()

def trajopt_shelves_demo():
    meshcat = StartMeshcat()
    meshcat.Delete()
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    iiwa = AddPlanarIiwa(plant)
    wsg = AddWsg(plant, iiwa, roll=0.0, welded=True, sphere=True)
    X_WStart = RigidTransform([0.8, 0, 0.65])
    meshcat.SetObject("start", Sphere(0.02), rgba=Rgba(.9, .1, .1, 1))
    meshcat.SetTransform("start", X_WStart)
    X_WGoal = RigidTransform([0.8, 0, 0.4])
    meshcat.SetObject("goal", Sphere(0.02), rgba=Rgba(.1, .9, .1, 1))
    meshcat.SetTransform("goal", X_WGoal)

    parser = Parser(plant)
    bin = parser.AddModelFromFile(
        FindResource("models/shelves.sdf"))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("shelves_body", bin),
                     RigidTransform([0.88, 0, 0.4]))
    plant.Finalize()

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration))
    collision_visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(prefix="collision", role=Role.kProximity))
    meshcat.SetProperty("collision", "visible", False)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    num_q = plant.num_positions()
    print('plant dimensionality :', num_q);

    q0 = plant.GetPositions(plant_context)
    gripper_frame = plant.GetFrameByName("body", wsg)

    trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 9)
    prog = trajopt.get_mutable_prog()
    trajopt.AddDurationCost(100.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(),
                              plant.GetPositionUpperLimits())
    trajopt.AddVelocityBounds(plant.GetVelocityLowerLimits(),
                              plant.GetVelocityUpperLimits())

    trajopt.AddDurationConstraint(.5, 5)

    # start constraint
    start_constraint = PositionConstraint(plant, plant.world_frame(),
                                          X_WStart.translation(),
                                          X_WStart.translation(), gripper_frame,
                                          [0, 0.1, 0], plant_context)
    trajopt.AddPathPositionConstraint(start_constraint, 0)
    prog.AddQuadraticErrorCost(np.eye(num_q), q0,
                               trajopt.control_points()[:, 0])

    # goal constraint
    goal_constraint = PositionConstraint(plant, plant.world_frame(),
                                         X_WGoal.translation(),
                                         X_WGoal.translation(), gripper_frame,
                                         [0, 0.1, 0], plant_context)
    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    prog.AddQuadraticErrorCost(np.eye(num_q), q0,
                               trajopt.control_points()[:, -1])

    # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 1)

    # Solve once without the collisions and set that as the initial guess for
    # the version with collisions.
    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed, even without collisions!")
        print(result.get_solver_id().name())
    else:
        print("Trajectory optimization succeded, when without collisions!")
        print(result.get_solver_id().name())
        opt_sol = result.GetSolution()
        print(' - opt_sol:', opt_sol.shape, dir(result))

    trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))


    # collision constraints
    collision_constraint = MinimumDistanceConstraint(plant, 0.001,
                                                     plant_context, None, 0.01)
    evaluate_at_s = np.linspace(0, 1, 25)
    for s in evaluate_at_s:
        trajopt.AddPathPositionConstraint(collision_constraint, s)

    def PlotPath(control_points):
        traj = BsplineTrajectory(trajopt.basis(),
                                 control_points.reshape((3, -1)))
        meshcat.SetLine('positions_path',
                         traj.vector_values(np.linspace(0, 1, 50)))

    print('trajopt.control_points().shape:', trajopt.control_points().shape)

    prog.AddVisualizationCallback(PlotPath,
                                  trajopt.control_points().reshape((-1,)))
    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed")
        print(result.get_solver_id().name())
    else:
        print('constrained optimization succeeds')

    PublishPositionTrajectory(trajopt.ReconstructTrajectory(result), context,
                              plant, visualizer, meshcat)
    collision_visualizer.ForcedPublish(
        collision_visualizer.GetMyContextFromRoot(context))
    print('break line to view animation:')
    _ = sys.stdin.readline()

def AddPackagePaths(parser):
    # Remove once https://github.com/RobotLocomotion/drake/issues/10531 lands.
    parser.package_map().PopulateFromFolder(FindResource(""))
    parser.package_map().Add(
        "manipulation_station",
        os.path.join(GetDrakePath(),
                     "examples/manipulation_station/models"))
    parser.package_map().Add(
        "ycb",
        os.path.join(GetDrakePath(), "manipulation/models/ycb"))
    parser.package_map().Add(
        "wsg_50_description",
        os.path.join(GetDrakePath(),
                     "manipulation/models/wsg_50_description"))


def trajopt_bins_demo():
    meshcat = StartMeshcat()
    meshcat.Delete()
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    parser = Parser(plant)
    AddPackagePaths(parser)
    bin = parser.AddAllModelsFromFile(
        FindResource("models/two_bins_w_cameras.dmd.yaml"))
    iiwa = AddIiwa(plant, collision_model="with_box_collision")
    wsg = AddWsg(plant, iiwa, welded=True, sphere=True)
    X_WStart = RigidTransform([0.5, 0, 0.15])
    meshcat.SetObject("start", Sphere(0.02), rgba=Rgba(.9, .1, .1, 1))
    meshcat.SetTransform("start", X_WStart)
    X_WGoal = RigidTransform([0, -0.6, 0.15])
    meshcat.SetObject("goal", Sphere(0.02), rgba=Rgba(.1, .9, .1, 1))
    meshcat.SetTransform("goal", X_WGoal)

    plant.Finalize()

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration))
    collision_visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(prefix="collision", role=Role.kProximity))
    meshcat.SetProperty("collision", "visible", False)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    num_q = plant.num_positions()
    q0 = plant.GetPositions(plant_context)
    gripper_frame = plant.GetFrameByName("body", wsg)

    trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 4)
    prog = trajopt.get_mutable_prog()

    # 
    q_guess = np.tile(q0.reshape((7,1)), (1, trajopt.num_control_points()))
    q_guess[0,:] = np.linspace(0, -np.pi/2, trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    # Uncomment this to see the initial guess:
    #PublishPositionTrajectory(path_guess, context, plant, visualizer)

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(),
                              plant.GetPositionUpperLimits())
    trajopt.AddVelocityBounds(plant.GetVelocityLowerLimits(),
                              plant.GetVelocityUpperLimits())

    trajopt.AddDurationConstraint(.5, 50)

    # start constraint
    start_constraint = PositionConstraint(plant, plant.world_frame(),
                                          X_WStart.translation(),
                                          X_WStart.translation(), gripper_frame,
                                          [0, 0.1, 0], plant_context)
    trajopt.AddPathPositionConstraint(start_constraint, 0)
    prog.AddQuadraticErrorCost(np.eye(num_q), q0,
                               trajopt.control_points()[:, 0])

    # goal constraint
    goal_constraint = PositionConstraint(plant, plant.world_frame(),
                                         X_WGoal.translation(),
                                         X_WGoal.translation(), gripper_frame,
                                         [0, 0.1, 0], plant_context)
    trajopt.AddPathPositionConstraint(goal_constraint, 1)
    prog.AddQuadraticErrorCost(np.eye(num_q), q0,
                               trajopt.control_points()[:, -1])

    # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 1)

    # collision constraints
    collision_constraint = MinimumDistanceConstraint(plant, 0.001,
                                                     plant_context, None, 0.01)
    evaluate_at_s = np.linspace(0, 1, 50)
    for s in evaluate_at_s:
        trajopt.AddPathPositionConstraint(collision_constraint, s)

    result = Solve(prog)
    if not result.is_success():
        print("Trajectory optimization failed")
        print(result.get_solver_id().name())

    PublishPositionTrajectory(trajopt.ReconstructTrajectory(result), context,
                              plant, visualizer)
    collision_visualizer.ForcedPublish(
        collision_visualizer.GetMyContextFromRoot(context))
    print('break line to view animation:')
    _ = sys.stdin.readline()


if '__main__' == __name__:
    #trajopt_bins_demo()
    trajopt_shelves_demo()
