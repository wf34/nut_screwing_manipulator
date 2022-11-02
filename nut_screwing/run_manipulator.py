import argparse
import time
import sys

import numpy as np

from bazel_tools.tools.python.runfiles import runfiles

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.common.eigen_geometry import AngleAxis
from pydrake.geometry import Rgba

from pydrake.all import (
    DiagramBuilder,
    MeshcatVisualizer,
    Simulator,
    ContactVisualizer, ContactVisualizerParams
)

import nut_screwing.sim_helper as sh
import nut_screwing.differential_controller as diff_c
import nut_screwing.open_loop_controller as ol_c
import nut_screwing.state_monitor as sm

DIFF_IK = 'differential'
OPEN_IK = 'open_loop'

def get_manipuland_resource_path():
    manifest = runfiles.Create()
    return manifest.Rlocation("control_nut_screwing_manipulator/resources/bolt_and_nut.sdf")


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

def AddContactsSystem(meshcat, builder):
    cv_params = ContactVisualizerParams()
    cv_params.publish_period = 0.05
    cv_params.default_color = Rgba(0.5, 0.5, 0.5)
    cv_params.prefix = "py_visualizer"
    cv_params.delete_on_initialization_event = False
    cv_params.force_threshold = 0.2
    cv_params.newtons_per_meter = 50
    cv_params.radius = 0.001

    cv_vis = ContactVisualizer(meshcat=meshcat, params=cv_params)
    cv_system = builder.AddSystem(cv_vis)
    cv_system.set_name('contact_visualizer')
    return cv_system


def build_scene(meshcat, controller_type, log_destination):
    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation())
    station.SetupNutStation()

    manipuland_path = get_manipuland_resource_path()
    cv_system = AddContactsSystem(meshcat, builder)
    station.Finalize()
    
    body_frames_visualization = False
    
    # Find the initial pose of the gripper (as set in the default Context)
    temp_context = station.CreateDefaultContext()
    plant = station.get_multibody_plant()
    
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    
    scene_graph = station.get_scene_graph()
    
    if body_frames_visualization:
        display_bodies_frames(plant, scene_graph)
            
    temp_st_context = station.GetMyContextFromRoot(temp_context)
    temp_plant_context = plant.GetMyContextFromRoot(temp_context)
    
    #q0 =[-1.56702176,  1.33784888,  0.00572793, -1.24946957, -0.002234,    2.05829444,
    #  0.00836547]
    #station.SetIiwaPosition(temp_st_context, q0)
    
    
    X_G = {"initial": plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("body"))}
    
    X_O = {"initial": RigidTransform(RotationMatrix(
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]),
        [0.0, -0.3, 0.1])}
    
    #X_O = {"initial": plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("nut"))}
    
    print('X_G[_initial]', X_G['initial'])
    print('X_O[_initial]', X_O['initial'])
    
    X_OinitialOgoal = RigidTransform(RotationMatrix.MakeZRotation(-np.pi / 6))
    X_O['goal'] = X_O['initial'].multiply(X_OinitialOgoal)
    #print(X_O, X_O['goal'], X_G)
    X_G, times = make_gripper_frames(X_G, X_O)

    if DIFF_IK == controller_type:
        input_iiwa_position_port = station.GetOutputPort("iiwa_position_measured")

        output_iiwa_position_port, output_wsg_position_port, integrator = \
             diff_c.create_differential_controller(builder, plant,
                                                   input_iiwa_position_port,
                                                   X_G, times)
    elif OPEN_IK == controller_type:
        integrator = None
        draw_frames = True
        output_iiwa_position_port, output_wsg_position_port, kfs, joint_space_trajectory = \
            ol_c.create_open_loop_controller(builder, plant, station,
                                             scene_graph, X_G, X_O,
                                             draw_frames)
    else:
        assert False, 'unreachable'

    builder.Connect(output_iiwa_position_port, station.GetInputPort("iiwa_position"))
    builder.Connect(output_wsg_position_port, station.GetInputPort("wsg_position"))
    builder.Connect(station.GetOutputPort("contact_results"), cv_system.contact_results_input_port())
    
    meshcat.Delete()
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)
    
    diagram = builder.Build()
    diagram.set_name("pick_adapted_to_nut")

    simulator = Simulator(diagram)
    state_monitor = sm.StateMonitor(log_destination, plant)
    simulator.set_monitor(state_monitor.callback)
    #station.SetIiwaPosition(station.GetMyContextFromRoot(simulator.get_mutable_context()), q0)
    
    if integrator is not None:
        station_context = station.GetMyContextFromRoot(simulator.get_mutable_context())
        # TODO(russt): Add this missing python binding
        #integrator.set_integral_value(
        #    integrator.GetMyContextFromRoot(simulator.get_mutable_context()), 
        #        station.GetIiwaPosition(station_context))
        integrator.GetMyContextFromRoot( \
            simulator.get_mutable_context()).get_mutable_continuous_state_vector() \
            .SetFromVector(station.GetIiwaPosition(station_context))
    
    simulator.set_target_realtime_rate(5.0)
    simulator.AdvanceTo(0.1)
    return simulator


def simulate_pick_adapted_to_nut(controller_type, log_destination):
    print('hello drake')
    meshcat = sh.StartMeshcat()
    simulator = build_scene(meshcat, controller_type, log_destination)
    print('break line to view animation:')
    _ = sys.stdin.readline()

    #meshcat.start_recording()
    simulator.set_target_realtime_rate(5.0)
    simulator.AdvanceTo(30)
    #meshcat.stop_recording()
    #meshcat.publish_recording()


def parse_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('-c', '--controller_type', required=True, choices=[DIFF_IK, OPEN_IK], help='what controls manipulator')
    parser.add_argument('-l', '--log_destination', default='', help='where to put telemetry')
    return vars(parser.parse_args())

if '__main__' == __name__:
    simulate_pick_adapted_to_nut(**parse_args())
