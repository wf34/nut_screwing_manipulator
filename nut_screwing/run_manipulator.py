import argparse
import time
import os
import sys

import numpy as np

import pydot

from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.common.eigen_geometry import AngleAxis
from pydrake.multibody.tree import RevoluteJoint_
from pydrake.geometry import Rgba
from pydrake.systems.primitives import Adder, ConstantVectorSource, Demultiplexer, PassThrough

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseDynamicsController,
    MeshcatVisualizer,
    MultibodyPlant,
    Simulator,
    ContactVisualizer, ContactVisualizerParams,
    StateInterpolatorWithDiscreteDerivative,
    SchunkWsgPositionController,
    MakeMultibodyStateToWsgStateSystem,
)
from pydrake.multibody.parsing import Parser

import sim_helper as sh
import differential_controller as diff_c
import new_differential_controller as diff2_c
import open_loop_controller as ol_c
import experimental_controller as e_c
import state_monitor as sm
import run_alt_manipulator as ram

EXPERIMENTAL = 'experimental'
DIFF_IK = 'differential'
DIFF2_IK = 'differential2'
OPEN_IK = 'open_loop'

TIME_STEP=0.0001 #finer
TIME_STEP=0.001  #orig
TIME_STEP=0.007  #faster

def get_manipuland_resource_path():
    #manifest = runfiles.Create()
    #. manifest.Rlocation
    proj_dir = os.environ.get('CNSM_PATH')
    assert proj_dir and os.path.isdir(proj_dir), proj_dir
    return os.path.join(proj_dir, 'resources/bolt_and_nut.sdf')


def add_manipuland(plant):
    manipuland_path = get_manipuland_resource_path()
    bolt_with_nut = Parser(plant=plant).AddModelFromFile(manipuland_path)
    X_WC = RigidTransform(RotationMatrix.Identity(), [0.0, -0.3, 0.1])
    plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName('bolt', bolt_with_nut),
            X_WC)
    return bolt_with_nut


def set_iiwa_default_position(plant, iiwa_model_name='iiwa7'):
    iiwa_model_instance = plant.GetModelInstanceByName(iiwa_model_name)
    indices = plant.GetJointIndices(model_instance=iiwa_model_instance)
    q0_iiwa = e_c.IIWA_DEFAULT_POSITION
    for i, q in zip(indices, q0_iiwa):
        ith_rev_joint = plant.get_mutable_joint(joint_index=i)
        assert isinstance(ith_rev_joint, RevoluteJoint_[float])
        ith_rev_joint.set_default_angle(q)


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





def AddContactsSystem(builder, plant, meshcat):
    cv_params = ContactVisualizerParams()
    #cv_params.publish_period = 0.05
    #cv_params.default_color = Rgba(0.5, 0.5, 0.5)
    cv_params.force_threshold = 1e-6
    cv_params.newtons_per_meter = 1.0
    cv_params.radius = 0.002
    ContactVisualizer.AddToBuilder(builder, plant, meshcat, cv_params)


def create_iiwa_position_measured_port(builder, plant, iiwa):
    num_iiwa_positions = plant.num_positions(iiwa)
    iiwa_output_state = plant.get_state_output_port(iiwa)
    demux = builder.AddSystem(Demultiplexer(size=num_iiwa_positions*2,
                                            output_ports_size=num_iiwa_positions))
    builder.Connect(plant.get_state_output_port(iiwa),
                    demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "iiwa_position_measured")

    controller_plant = MultibodyPlant(time_step=TIME_STEP)
    controller_iiwa = ram.AddIiwa(controller_plant, collision_model="with_box_collision")
    ram.AddWsg(controller_plant, controller_iiwa, welded=True, sphere=True)
    controller_plant.Finalize()

    iiwa_controller = builder.AddSystem(
                InverseDynamicsController(controller_plant,
                                          kp=[100] * num_iiwa_positions,
                                          ki=[1] * num_iiwa_positions,
                                          kd=[20] * num_iiwa_positions,
                                          has_reference_acceleration=False))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(plant.get_state_output_port(iiwa),
                    iiwa_controller.get_input_port_estimated_state())

    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    torque_passthrough = builder.AddSystem(PassThrough([0] * num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(), "iiwa_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(iiwa))

    return demux.get_output_port(0), iiwa_controller, iiwa_output_state


def create_iiwa_position_desired_port(builder, plant, iiwa, iiwa_pid_controller):
    num_iiwa_positions = plant.num_positions(iiwa)
    desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_iiwa_positions,
                    TIME_STEP,
                    suppress_initial_transient=True))
    desired_state_from_position.set_name("iiwa_desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(),
                    iiwa_pid_controller.get_input_port_desired_state())

    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
    #builder.ExportInput(iiwa_position.get_input_port(), "iiwa_position")
    builder.ExportOutput(iiwa_position.get_output_port(), "iiwa_position_commanded")
    builder.Connect(iiwa_position.get_output_port(), desired_state_from_position.get_input_port())
    return iiwa_position.get_input_port()


def create_wsg_position_desired_port(builder, plant, wsg):
    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    wsg_controller.set_name("wsg_controller")
    builder.Connect(wsg_controller.get_generalized_force_output_port(),
                    plant.get_actuation_input_port(wsg))
    builder.Connect(plant.get_state_output_port(wsg),
                    wsg_controller.get_state_input_port())
    #builder.ExportInput(
    #    wsg_controller.get_desired_position_input_port(),
    #    "wsg_position")

    wsg_mbp_state_to_wsg_state = builder.AddSystem(MakeMultibodyStateToWsgStateSystem())
    builder.Connect(plant.get_state_output_port(wsg), wsg_mbp_state_to_wsg_state.get_input_port())

    return wsg_controller.get_desired_position_input_port()


def build_scene(meshcat, controller_type, log_destination):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    iiwa = ram.AddIiwa(plant, collision_model="with_box_collision")
    wsg = ram.AddWsg(plant, iiwa, welded=False, sphere=True)

    #station.SetupManipulationClassStation()  # .SetupNutStation()
    #plant = station.get_multibody_plant()

    bolt_with_nut = add_manipuland(plant)
    zero_torque_system = builder.AddSystem(ConstantVectorSource(np.zeros(1)))

    plant.Finalize()
    # AddContactsSystem(builder, plant, meshcat)

    nut_input_port = plant.get_actuation_input_port(model_instance=bolt_with_nut)
    iiwa_actuation_input_port = plant.get_actuation_input_port(model_instance=iiwa)

    set_iiwa_default_position(plant)
    body_frames_visualization = False

    
    if body_frames_visualization:
        display_bodies_frames(plant, scene_graph)

    #X_G = {"initial":
    #   RigidTransform(
    #    R=RotationMatrix([
    #        [0.9999996829318348, 0.00019052063137842194, -0.0007731999219133522],
    #        [0.0007963267107334455, -0.23924925335563643, 0.9709578572896668],
    #        [1.868506971441006e-16, -0.9709581651495911, -0.23924932921398248],
    #      ]),
    #      p=[0.0003753557139120804, -0.4713587901037817, 0.6560185829424987]
    #    )
    #}

    #X_O = {"initial": RigidTransform(RotationMatrix(
    #    [[1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 1.0]]),
    #     [0.0, -0.3, 0.1])}

    #X_O = {"initial": plant.EvalBodyPoseInWorld(temp_plant_context, plant.GetBodyByName("nut"))}
    
    #X_OinitialOgoal = RigidTransform(RotationMatrix.MakeZRotation(-np.pi / 6))
    #X_O['goal'] = X_O['initial'].multiply(X_OinitialOgoal)
    #X_G, times = diff2_c.make_gripper_frames(X_G, X_O)
    #print('prepick at {}; pick_start at {}, ends at {}.'.format(
    #    times['prepick'], times['pick_start'], times['pick_end']))

    measured_iiwa_position_port, iiwa_pid_controller, measured_iiwa_state_port = \
        create_iiwa_position_measured_port(
            builder, plant, iiwa)
    desired_iiwa_position_port = create_iiwa_position_desired_port(builder, plant, iiwa, iiwa_pid_controller)
    desired_wsg_position_port = create_wsg_position_desired_port(builder, plant, wsg)

    if DIFF_IK == controller_type:
        output_iiwa_position_port, output_wsg_position_port, integrator = \
             diff_c.create_differential_controller(builder, plant,
                                                   measured_iiwa_position_port,
                                                   X_G, times)
    elif DIFF2_IK == controller_type:
        integrator = None
        output_iiwa_position_port, output_wsg_position_port = \
            diff2_c.add_new_differential_controller(builder, plant, measured_iiwa_state_port, iiwa_pid_controller, meshcat)
    elif OPEN_IK == controller_type:
        integrator = None
        draw_frames = True
        output_iiwa_position_port, output_wsg_position_port, kfs, joint_space_trajectory = \
            ol_c.create_open_loop_controller(builder, plant,
                                             scene_graph, X_G, X_O,
                                             draw_frames)
    elif EXPERIMENTAL == controller_type:
        print('makes', EXPERIMENTAL)
        output_iiwa_position_port, output_wsg_position_port, integrator = \
            e_c.create_experimental_controller(builder, plant, measured_iiwa_position_port,
                                               temp_context, X_G)
    else:
        assert False, 'unreachable'

    if not output_iiwa_position_port or not output_wsg_position_port:
        print('controller has failed')
        return

    builder.Connect(output_iiwa_position_port, desired_iiwa_position_port)
    builder.Connect(output_wsg_position_port, desired_wsg_position_port)

    # builder.Connect(plant.get_contact_results_output_port(), cv_system.contact_results_input_port())
    builder.Connect(zero_torque_system.get_output_port(0), nut_input_port)

    #meshcat.Delete()

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat)
    diagram = builder.Build()
    diagram.set_name("nut_screwing")

    pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].write_png('diagram.png')

    simulator = Simulator(diagram)
    if log_destination != '':
        state_monitor = sm.StateMonitor(log_destination, plant)
        simulator.set_monitor(state_monitor.callback)

    #plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    #plant.SetPositions(plant_context, iiwa, e_c.IIWA_DEFAULT_POSITION)
    #print('--------- X_G:\n', plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body")))
    # Find the initial pose of the gripper (as set in the default Context)
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    simulator.Initialize()

    if integrator is not None:
        integrator.set_integral_value(
            integrator.GetMyContextFromRoot(simulator.get_mutable_context()),
                plant.GetPositions(plant.GetMyContextFromRoot(simulator.get_mutable_context()),
                                   plant.GetModelInstanceByName("iiwa7")))
    
    return simulator, visualizer


def simulate_nut_screwing(controller_type, log_destination):
    print('hello drake')
    meshcat = sh.StartMeshcat()
    simulator, visualizer = build_scene(meshcat, controller_type, log_destination)


    web_url = meshcat.web_url()
    print(f'Meshcat is now available at {web_url}')
    os.system(f'xdg-open {web_url}')

    if not simulator:
        os.sleep(5.)
        return

    visualizer.StartRecording(False)
    simulator.AdvanceTo(40)
    visualizer.PublishRecording()
    time.sleep(30)


def parse_args():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('-c', '--controller_type', required=True, choices=[DIFF_IK, DIFF2_IK, OPEN_IK, EXPERIMENTAL], help='what controls manipulator')
    parser.add_argument('-l', '--log_destination', default='', help='where to put telemetry')
    return vars(parser.parse_args())

if '__main__' == __name__:
    simulate_nut_screwing(**parse_args())
