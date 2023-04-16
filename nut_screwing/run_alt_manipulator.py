import argparse
import functools
import itertools
import time
import os
import sys

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.all import (AddMultibodyPlantSceneGraph, BsplineTrajectory, GeometryInstance,
                         DiagramBuilder, KinematicTrajectoryOptimization, LinearConstraint,
                         MeshcatVisualizer, MeshcatVisualizerParams,
                         MinimumDistanceConstraint, Parser, PositionConstraint, OrientationConstraint,
                         Rgba, RigidTransform, Role, Solve, Sphere,
                         StartMeshcat, FindResourceOrThrow, RevoluteJoint, RollPitchYaw, GetDrakePath, MeshcatCone,
                         ConstantVectorSource, StackedTrajectory)
from pydrake.geometry import (Cylinder, GeometryInstance,
                              MakePhongIllustrationProperties)
from pydrake.multibody import inverse_kinematics

from experimental_controller import IIWA_DEFAULT_POSITION
import sim_helper as sh

import run_manipulator as rm
import new_differential_controller as diff2_c
from experimental_controller import get_default_plant_position_with_inf

SHELVES = 'shelves'
BINS = 'bins'
TRAJOPT_SCREWING = 'trajopt_screwing'
SCENARIOS = [SHELVES, BINS, TRAJOPT_SCREWING]


def string_from_transform(X):
    data = X.translation().tolist() + [np.degrees(X.rotation().ToRollPitchYaw().vector()[-1])]
    return 'pos={:.2f}, {:.2f}, {:.2f}; azimuth={:.0f}'.format(*data)


def handle_opt_result(result, trajopt, prog):
    if not result.is_success():
        print("Trajectory optimization failed")
        print(dir(result))
        print(result.get_solver_id().name(), result.GetInfeasibleConstraintNames(prog))
    else:
        print("Trajectory optimization succeeded")
        return trajopt.ReconstructTrajectory(result)


def get_present_plant_position_with_inf(plant, plant_context, information=1., model_name='iiwa'):
    iiwa_model_instance = plant.GetModelInstanceByName(model_name)
    indices = list(map(int, plant.GetJointIndices(model_instance=iiwa_model_instance)))
    n = plant.num_positions()
    plant_0 = plant.GetPositions(plant_context)
    plant_inf = np.eye(n) * 1.e-9
    for i in zip(indices):
        plant_inf[i, i] = 1
    return plant_0, plant_inf


def AddTriad(source_id,
             frame_id,
             scene_graph,
             length=.25,
             radius=0.01,
             opacity=1.,
             X_FT=RigidTransform(),
             name="frame"):
    """
    Adds illustration geometry representing the coordinate frame, with the
    x-axis drawn in red, the y-axis in green and the z-axis in blue. The axes
    point in +x, +y and +z directions, respectively.

    Args:
      source_id: The source registered with SceneGraph.
      frame_id: A geometry::frame_id registered with scene_graph.
      scene_graph: The SceneGraph with which we will register the geometry.
      length: the length of each axis in meters.
      radius: the radius of each axis in meters.
      opacity: the opacity of the coordinate axes, between 0 and 1.
      X_FT: a RigidTransform from the triad frame T to the frame_id frame F
      name: the added geometry will have names name + " x-axis", etc.
    """
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2),
                          [length / 2., 0, 0])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length),
                            name + " x-axis")
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([1, 0, 0, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2),
                          [0, length / 2., 0])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length),
                            name + " y-axis")
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 1, 0, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length),
                            name + " z-axis")
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 0, 1, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)


def AddMultibodyTriad(frame, scene_graph, length=.25, radius=0.01, opacity=1.):
    plant = frame.GetParentPlant()
    AddTriad(plant.get_source_id(),
             plant.GetBodyFrameIdOrThrow(frame.body().index()), scene_graph,
             length, radius, opacity, frame.GetFixedPoseInBodyFrame())


def display_bodies_frames(plant, scene_graph):
    for body_name in ["body", "nut"]:
        AddMultibodyTriad(plant.GetFrameByName(body_name), scene_graph)


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

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, np.pi / 2.0), [0, 0, 0.114])
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
    q0 = IIWA_DEFAULT_POSITION
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


def AddMeshactProgressSphere(meshcat, current_time, total_duration, plant, root_context):
    plant_context = plant.GetMyContextFromRoot(root_context)

    blue = np.array([0., 0., 1., 1.])
    green = np.array([0., 1., 0., 1.])

    a = current_time / total_duration
    assert 0. <= a and a <= 1.
    b = 1. - a
    mixture = a * blue + b * green

    root_context.SetTime(current_time)

    X_W_G = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))
    curr_point = 'point_{}'.format(current_time)
    meshcat.SetObject(curr_point, Sphere(0.01), rgba=Rgba(*mixture.tolist()))
    meshcat.SetTransform(curr_point, X_W_G)


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

    dur = trajectory.end_time()  - trajectory.start_time()
    for t in np.append(
            np.arange(trajectory.start_time(), trajectory.end_time(),
                      time_step), trajectory.end_time()):
        plant.SetPositions(plant_context, trajectory.value(t))
        if meshcat:
            AddMeshactProgressSphere(meshcat, t, dur, plant, root_context)
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()


def PublishStackOfPositionTrajectores(trajectories,
                                      root_context,
                                      plant,
                                      visualizer,
                                      meshcat=None,
                                      time_step=1.0 / 33.0):
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)
    visualizer.StartRecording(False)

    ends = list(itertools.accumulate(map(lambda x: x.end_time(), trajectories)))
    overall_length = functools.reduce(lambda x, y: x + y, ends, 0)
    ends = [0.] + ends[:-1]

    current_traj_index = 0
    for t in np.append(np.arange(0., overall_length, time_step), overall_length):
        current_time = t - ends[current_traj_index]
        if current_time > trajectories[current_traj_index].end_time():
            current_traj_index += 1

        if current_traj_index == len(trajectories):
            visualizer.ForcedPublish(visualizer_context)
            break

        trajectory = trajectories[current_traj_index]
        plant.SetPositions(plant_context, trajectory.value(current_time))
        if meshcat:
            AddMeshactProgressSphere(meshcat, t, overall_length, plant, root_context)
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()


def trajopt_shelves_demo(meshcat):
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


def trajopt_bins_demo(meshcat):
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

    trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 10)
    prog = trajopt.get_mutable_prog()

    q_guess = np.tile(q0.reshape((7,1)), (1, trajopt.num_control_points()))
    q_guess[0,:] = np.linspace(0, -np.pi/2, trajopt.num_control_points())
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    # Uncomment this to see the initial guess:
    # PublishPositionTrajectory(path_guess, context, plant, visualizer)

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(),
                              plant.GetPositionUpperLimits())
    trajopt.AddVelocityBounds(plant.GetVelocityLowerLimits(),
                              plant.GetVelocityUpperLimits())

    trajopt.AddDurationConstraint(5, 50)

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
    else:
        print("Trajectory optimization succeeded")

    print('break line to view animation:')
    _ = sys.stdin.readline()
    PublishPositionTrajectory(trajopt.ReconstructTrajectory(result), context,
                              plant, visualizer)
    collision_visualizer.ForcedPublish(
        collision_visualizer.GetMyContextFromRoot(context))


def parse_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('-s', '--scenario', default=TRAJOPT_SCREWING, choices=SCENARIOS, help='which scene is modeled')
    return vars(parser.parse_args())


def constrain_position(plant, trajopt,
                       X_WG, target_time,
                       plant_context,
                       with_orientation=False,
                       pos_limit=0.0):
    lower_translation, upper_translation = \
        X_WG.translation() - [pos_limit]*3, X_WG.translation() + [pos_limit]*3

    gripper_frame = plant.GetBodyByName("body").body_frame()
    pos_constraint = PositionConstraint(plant, plant.world_frame(),
                                        lower_translation,
                                        upper_translation,
                                        gripper_frame,
                                        [0, 0., 0.],
                                        plant_context)
    trajopt.AddPathPositionConstraint(pos_constraint, target_time)

    if with_orientation:
        orientation_constraint = OrientationConstraint(plant,
                                                       gripper_frame,
                                                       X_WG.rotation().inverse(),
                                                       plant.world_frame(),
                                                       RotationMatrix(),
                                                       np.radians(5),
                                                       plant_context)
        trajopt.AddPathPositionConstraint(orientation_constraint, target_time)


def get_torque_coords(plant, X_WGgoal, q0):
    ik = inverse_kinematics.InverseKinematics(plant)
    q_variables = ik.q()
    # print(q_variables.shape, q0.shape)
    prog = ik.prog()
    prog.SetInitialGuess(q_variables, q0)
    prog.AddCost(np.square(np.dot(q_variables, q0)))
    ik.AddPositionConstraint(
        frameA=plant.GetFrameByName("body"),
        frameB=plant.world_frame(),
        p_BQ=X_WGgoal.translation(),
        p_AQ_lower=[-0.02, -0.02, -0.05], p_AQ_upper=[0.02, 0.02, 0.05])
    ik.AddOrientationConstraint(
        frameAbar=plant.GetFrameByName("body"),
        R_AbarA=X_WGgoal.rotation().inverse(), 
        frameBbar=plant.world_frame(),
        R_BbarB=RotationMatrix(),
        theta_bound=np.radians(3))
    result = Solve(prog)
    assert result.is_success()
    # print(result.get_solver_id().name())
    return result.GetSolution(q_variables)


def run_traj_opt_towards_prepick(traj_name,
                                 X_WGStart, X_WGgoal,
                                 plant, context,
                                 visualizer=None):
    plant_context = plant.GetMyContextFromRoot(context)
    print('optimize [{}] {} from {}'.format(
        traj_name,
        string_from_transform(X_WGgoal),
        string_from_transform(X_WGStart)))

    num_q = plant.num_positions()
    num_c = 12
    # print('num_positions: {}; num control points: {}'.format(num_q, num_c))

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)

    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())

    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0)
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0)
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddDurationConstraint(5, 10)

    start_lim = 0
    end_lim   = 0.03

    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=False, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal,  1, plant_context,
                       with_orientation=True, pos_limit=end_lim)

    q0, inf0 = get_present_plant_position_with_inf(plant, plant_context, 1., 'iiwa7')
    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, 0])
    prog.AddQuadraticErrorCost(inf0, q0, trajopt.control_points()[:, -1])

    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog)


def run_traj_opt_towards_pick(traj_name,
                              X_WGStart, X_WGgoal,
                              plant, context,
                              visualizer=None):
    plant_context = plant.GetMyContextFromRoot(context)
    print('optimize [{}] {} from {}'.format(
        traj_name,
        string_from_transform(X_WGgoal),
        string_from_transform(X_WGStart)))

    num_q = plant.num_positions()
    num_c = 12

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(),
                              plant.GetPositionUpperLimits())
    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0)
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0)
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddDurationConstraint(1, 5)
    qx, inf0 = get_present_plant_position_with_inf(plant, plant_context, 1., 'iiwa7')
    q_goal = get_torque_coords(plant, X_WGgoal, qx)

    #print('qx', qx)
    #print('qgoal', q_goal)

    #plant.SetPositions(plant_context, q_goal)
    #visualizer.ForcedPublish(visualizer.GetMyContextFromRoot(context))
    #time.sleep(30)
    #exit(0)

    assert len(qx) == num_q
    assert len(q_goal) == num_q
    q_guess = np.linspace(qx.reshape((num_q, 1)),
                          q_goal.reshape((num_q, 1)),
                          trajopt.num_control_points()
        )[:, :, 0].T
    trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))

    start_lim = 0
    end_lim   = 0.03
    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal,  1, plant_context,
                       with_orientation=True, pos_limit=end_lim)

    prog.AddQuadraticErrorCost(inf0, qx, trajopt.control_points()[:, 0])
    prog.AddQuadraticErrorCost(inf0, q_goal, trajopt.control_points()[:, -1])

    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog)


def run_traj_opt_towards_place(traj_name,
                               X_WGStart, X_WGgoal,
                               plant, context,
                               visualizer=None):
    plant_context = plant.GetMyContextFromRoot(context)
    print('optimize [{}] {} from {}'.format(
        traj_name,
        string_from_transform(X_WGgoal),
        string_from_transform(X_WGStart)))

    num_q = plant.num_positions()
    num_c = 12
    print('num_positions: {}; num control points: {}'.format(num_q, num_c))

    trajopt = KinematicTrajectoryOptimization(num_q, num_c)
    prog = trajopt.get_mutable_prog()

    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(),
                              plant.GetPositionUpperLimits())
    plant_v_lower_limits = np.nan_to_num(plant.GetVelocityLowerLimits(), neginf=0)
    plant_v_upper_limits = np.nan_to_num(plant.GetVelocityUpperLimits(), posinf=0)
    trajopt.AddVelocityBounds(plant_v_lower_limits, plant_v_upper_limits)

    trajopt.AddDurationConstraint(5, 10)
    qx, inf0 = get_present_plant_position_with_inf(plant, plant_context, 1., 'iiwa7')

    start_lim = 0
    end_lim   = 0.03
    constrain_position(plant, trajopt, X_WGStart, 0, plant_context,
                       with_orientation=True, pos_limit=start_lim)
    constrain_position(plant, trajopt, X_WGgoal,  1, plant_context,
                       with_orientation=True, pos_limit=end_lim)

    q_goal = get_torque_coords(plant, X_WGgoal, qx)

    #plant.SetPositions(plant_context, q_goal)
    #visualizer.ForcedPublish(visualizer.GetMyContextFromRoot(context))
    #time.sleep(30)
    #exit(0)

    q_guess = np.linspace(qx.reshape((num_q, 1)),
                          q_goal.reshape((num_q, 1)),
                          trajopt.num_control_points()
        )[:, :, 0].T
    trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))

    prog.AddQuadraticErrorCost(inf0, qx, trajopt.control_points()[:, 0])
    prog.AddQuadraticErrorCost(inf0, q_goal, trajopt.control_points()[:, -1])

    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros(
        (num_q, 1)), 1)

    result = Solve(prog)
    return handle_opt_result(result, trajopt, prog)

def trajopt_screwing_demo(meshcat):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=rm.TIME_STEP)
    iiwa = AddIiwa(plant, collision_model="with_box_collision")
    wsg = AddWsg(plant, iiwa, welded=False, sphere=False)

    bolt_with_nut = rm.add_manipuland(plant)
    zero_torque_system = builder.AddSystem(ConstantVectorSource(np.zeros(1)))
    plant.Finalize()

    display_bodies_frames(plant, scene_graph)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration))
    collision_visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(prefix="collision", role=Role.kProximity))
    meshcat.SetProperty("collision", "visible", False)

    diagram = builder.Build()
    diagram.set_name("nut_screwing")

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    rm.set_iiwa_default_position(plant)
    
    gripper_body_index = int(plant.GetBodyByName("body").index())
    nut_body_index = int(plant.GetBodyByName("nut").index())
    X_WBcurrent_getter = lambda body_index: plant.get_body_poses_output_port().Eval(plant_context)[body_index]
    X_WGcurrent_getter = lambda _=None: X_WBcurrent_getter(gripper_body_index)

    X_G = { "initial": X_WGcurrent_getter() }
    X_O = { "initial": X_WBcurrent_getter(nut_body_index) }

    X_OinitialOgoal = RigidTransform(RotationMatrix.MakeZRotation(-np.pi / 6))
    X_O['goal'] = X_O['initial'].multiply(X_OinitialOgoal)
    X_G, times = diff2_c.make_gripper_frames(X_G, X_O)

    diff2_c.AddMeshcatTriad(meshcat, 'start', X_PT=X_G['initial'], opacity=0.2)

    goal_frames = ['prepick', 'pick', 'place']
    names = {
        'prepick' : 'prepick from init',
        'pick' : 'pick from prepick',
        'place' : 'place from pick',
    }
    funcs = {
        'prepick' : run_traj_opt_towards_prepick,
        'pick' : run_traj_opt_towards_pick,
        'place' : run_traj_opt_towards_place,
    }

    stacked_trajectores = []
    for goal in goal_frames:
        X_WGStart = X_WGcurrent_getter()
        X_WGgoal = X_G[goal]
        diff2_c.AddMeshcatTriad(meshcat, 'goal_{}'.format(goal), X_PT=X_WGgoal, opacity=0.4)

        f = funcs[goal]
        interm_traj = f(names[goal], X_WGStart, X_WGgoal, plant, context, visualizer)
        if not interm_traj:
            return

        plant.SetPositions(plant_context, interm_traj.FinalValue())
        visualizer.ForcedPublish(visualizer.GetMyContextFromRoot(context))
        stacked_trajectores.append(interm_traj)

    PublishStackOfPositionTrajectores(stacked_trajectores, context, plant, visualizer, meshcat)
    collision_visualizer.ForcedPublish(collision_visualizer.GetMyContextFromRoot(context))


def run_alt_main(scenario):
    meshcat = sh.StartMeshcat()
    web_url = meshcat.web_url()
    print(f'Meshcat is now available at {web_url}')
    os.system(f'xdg-open {web_url}')
    if scenario == SHELVES:
        trajopt_shelves_demo(meshcat)
    elif scenario == BINS:
        trajopt_bins_demo(meshcat)
    elif scenario == TRAJOPT_SCREWING:
        trajopt_screwing_demo(meshcat)
    print('python sent to sleep')
    time.sleep(30)


if '__main__' == __name__:
    run_alt_main(**parse_args())
