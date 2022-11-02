import argparse
import time
import sys

import numpy as np

from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.common.eigen_geometry import AngleAxis
from pydrake.geometry import Rgba
from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    MeshcatVisualizer,
    Simulator,
    ContactVisualizer, ContactVisualizerParams,
    SpatialForce, ExternallyAppliedSpatialForce,
    ConstantValueSource
)


import nut_screwing.sim_helper as sh
import nut_screwing.differential_controller as diff_c
import nut_screwing.open_loop_controller as ol_c
import nut_screwing.state_monitor as sm

DIFF_IK = 'differential'
OPEN_IK = 'open_loop'

def AddExternallyAppliedSpatialForce(builder, station):
    force_object = ExternallyAppliedSpatialForce()
    force_object.body_index = station.get_multibody_plant().GetBodyByName("nut").index()
    force_object.p_BoBq_B = np.array([0.02, 0.02, 0.])
    #force_object.F_Bq_W = SpatialForce(tau=np.array([0., 0., -3000.]), f=np.zeros(3))
    force_object.F_Bq_W = SpatialForce(tau=np.zeros(3), f=np.array([200., 1., 0.]))

    forces = []
    forces.append(force_object)
    value = AbstractValue.Make(forces)
    force_system = builder.AddSystem(ConstantValueSource(value))
    force_system.set_name('constant_debug_force')
    return force_system


def make_gripper_frames(X_G, X_O):
    """
    Takes a partial specification with X_G["initial"] and X_O["initial"] and X_0["goal"], and 
    returns a X_G and times with all of the pick and place frames populated.
    """

    assert 'initial' in X_G
    assert 'initial' in X_O
    #assert 'goal' in X_O
    # Define (again) the gripper pose relative to the object when in grasp.
    
    p_GgraspO = [0., 0.3, 0.10] # I want to achieve this version
    #p_GgraspO = [0.0, 0.1, 0.07] # which of these to use depends on gravity
    R_GgraspO = RotationMatrix.MakeZRotation(np.pi/6.0)
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    
    X_OGgrasp = X_GgraspO.inverse()

    X_GpregraspGgrasp = RigidTransform(RotationMatrix.Identity(), [0.1, 0., 0.])
    X_GgraspGpregrasp = X_GpregraspGgrasp.inverse()


    p_GpickGmoved_through = [-0.1, 0.2, 0.0]
    X_GpickGmoved_through = RigidTransform(RotationMatrix.MakeZRotation(np.radians(5.)), p_GpickGmoved_through)

    X_G["pick_start"] = X_O["initial"].multiply(X_OGgrasp)
    X_G["pregrasp"] = X_G["pick_start"].multiply(X_GgraspGpregrasp)
    X_G["pick_end"] = X_G["pick_start"].multiply(X_GpickGmoved_through)
    X_G["pull_out"] = X_G["pregrasp"]

    # Now let's set the timing
    times = {"initial": 0}
    times["pregrasp"] = times["initial"] + 4.1
    times["pick_start"] = times["pregrasp"] + 1.
    times["pick_end"] = times["pick_start"] + 0.5
    times["pull_out"] = times["pick_end"] + 1.
    
    for step in ['initial', 'pregrasp', 'pick_start', 'pick_end']:
        three_to_str = lambda y : ' '.join(map(lambda x: '{:.3f}'.format(x), y))
        rpy = RollPitchYaw(X_G[step].rotation()).vector()
        tr = X_G[step].translation()

        print('Step {}. rpy: {}; tr: {} \n'.format(step, three_to_str(rpy), three_to_str(tr)))
    #exit(1)

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


def build_scene(meshcat, controller_type, log_destination, with_external_force):
    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation())
    station.SetupNutStation()
    cv_system = AddContactsSystem(meshcat, builder)

    if with_external_force:
        force_system = AddExternallyAppliedSpatialForce(builder, station)

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
    #print(X_O)
    
    print('X_G[_initial]', X_G['initial'])
    print('X_O[_initial]', X_O['initial'])
    
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
    if with_external_force:
        builder.Connect(force_system.get_output_port(0), station.GetInputPort('applied_spatial_force'))
    
    meshcat.Delete()
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), meshcat)
    
    diagram = builder.Build()
    diagram.set_name("nut_screwing")

    simulator = Simulator(diagram)
    state_monitor = sm.StateMonitor(log_destination, plant)
    simulator.set_monitor(state_monitor.callback)
    #station.SetIiwaPosition(station.GetMyContextFromRoot(simulator.get_mutable_context()), q0)

    screw = plant.GetJointByName("nut_to_bolt_link")
    print(screw, screw.screw_pitch(), screw.damping(), screw.get_translation(temp_plant_context), screw.get_rotation(temp_plant_context))
    nut = plant.GetBodyByName("nut")
    print(type(nut), dir(nut))
    print('nut weighs: ', nut.get_mass(temp_plant_context))
    print('nut I: ', nut.default_rotational_inertia().get_moments(), nut.default_rotational_inertia().get_products())
    
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


def simulate_manipulator_scene(controller_type, log_destination, with_external_force):
    print('hello drake')
    meshcat = sh.StartMeshcat()
    simulator = build_scene(meshcat, controller_type, log_destination, with_external_force)
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
    parser.add_argument('-e', '--with_external_force', action='store_true', help='set if synthetic manipulation is needed')
    return vars(parser.parse_args())

if '__main__' == __name__:
    simulate_manipulator_scene(**parse_args())
