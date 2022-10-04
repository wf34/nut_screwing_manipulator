"""
Runs the manipulation_station example with a meshcat joint slider ui for
directly tele-operating the joints.  To have the meshcat server automatically
open in your browser, supply the --open-window flag; the joint sliders will be
accessible by clicking on "Open Controls" in the top right corner.
"""

import argparse
import sys
import webbrowser

import numpy as np

#from drake.examples.manipulation_station.schunk_wsg_buttons import \
#    SchunkWsgButtons
from pydrake.examples import (
    CreateClutterClearingYcbObjectList, ManipulationStation)
from pydrake.geometry import DrakeVisualizer
from pydrake.multibody.meshcat import JointSliders
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.geometry import Meshcat, MeshcatVisualizer
from pydrake.systems.primitives import FirstOrderLowPassFilter, VectorLogSink

from pydrake.systems.framework import LeafSystem


class SchunkWsgButtons(LeafSystem):
    """
    Adds buttons to open/close the Schunk WSG gripper.

    .. pydrake_system::

        name: SchunkWsgButtons
        output_ports:
        - position
        - max_force
    """

    _BUTTON_NAME = "Open/Close Gripper"
    """The name of the button added to the meshcat UI."""

    def __init__(self, meshcat, open_position=0.107, closed_position=0.002,
                 force_limit=40):
        """"
        Args:
            open_position:   Target position for the gripper when open.
            closed_position: Target position for the gripper when closed.
                             **Warning**: closing to 0mm can smash the fingers
                             together and keep applying force even when no
                             object is grasped.
            force_limit:     Force limit to send to Schunk WSG controller.
        """
        super().__init__()
        self.meshcat = meshcat
        self.DeclareVectorOutputPort("position", 1, self.CalcPositionOutput)
        self.DeclareVectorOutputPort("force_limit", 1,
                                     self.CalcForceLimitOutput)
        self._open_button = meshcat.AddButton(self._BUTTON_NAME)
        self._open_position = open_position
        self._closed_position = closed_position
        self._force_limit = force_limit

    def CalcPositionOutput(self, context, output):
        if self.meshcat.GetButtonClicks(name=self._BUTTON_NAME) % 2 == 0:
            output.SetAtIndex(0, self._open_position)
        else:
            output.SetAtIndex(0, self._closed_position)

    def CalcForceLimitOutput(self, context, output):
        output.SetAtIndex(0, self._force_limit)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Desired rate relative to real time.  See documentation for "
             "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument(
        "--duration", type=float, default=np.inf,
        help="Desired duration of the simulation in seconds.")
    parser.add_argument(
        "--test", action='store_true',
        help="Disable opening the gui window for testing.")
    parser.add_argument(
        "-w", "--open-window", dest="browser_new",
        action="store_const", const=1, default=None,
        help="Open the MeshCat display in a new browser window.")
    args = parser.parse_args()

    builder = DiagramBuilder()
    meshcat = Meshcat()

    station = builder.AddSystem(ManipulationStation())
    station.SetupNutStation()
    station.Finalize()

    geometry_query_port = station.GetOutputPort("geometry_query")
    DrakeVisualizer.AddToBuilder(builder, geometry_query_port)
    meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
        builder=builder,
        query_object_port=geometry_query_port,
        meshcat=meshcat)

    if args.browser_new is not None:
        url = meshcat.web_url()
        webbrowser.open(url=url, new=args.browser_new)

    teleop = builder.AddSystem(JointSliders(
        meshcat=meshcat, plant=station.get_controller_plant()))

    num_iiwa_joints = station.num_iiwa_joints()
    filter = builder.AddSystem(FirstOrderLowPassFilter(
        time_constant=1.0, size=num_iiwa_joints))
    builder.Connect(teleop.get_output_port(0), filter.get_input_port(0))
    builder.Connect(filter.get_output_port(0),
                    station.GetInputPort("iiwa_position"))

    wsg_buttons = builder.AddSystem(SchunkWsgButtons(meshcat=meshcat))
    builder.Connect(wsg_buttons.GetOutputPort("position"),
                    station.GetInputPort("wsg_position"))
    builder.Connect(wsg_buttons.GetOutputPort("force_limit"),
                    station.GetInputPort("wsg_force_limit"))

    # When in regression test mode, log our joint velocities to later check
    # that they were sufficiently quiet.
    if args.test:
        iiwa_velocities = builder.AddSystem(VectorLogSink(num_iiwa_joints))
        builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                        iiwa_velocities.get_input_port(0))
    else:
        iiwa_velocities = None

    diagram = builder.Build()
    simulator = Simulator(diagram)

    simulator.set_publish_every_time_step(False)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    station.GetInputPort("iiwa_feedforward_torque").FixValue(
        station_context, np.zeros(num_iiwa_joints))

    # Eval the output port once to read the initial positions of the IIWA.
    q0 = station.GetOutputPort("iiwa_position_measured").Eval(
        station_context)
    teleop.SetPositions(q0)
    filter.set_initial_output_value(diagram.GetMutableSubsystemContext(
        filter, simulator.get_mutable_context()), q0)

    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.AdvanceTo(args.duration)

    # Ensure that our initialization logic was correct, by inspecting our
    # logged joint velocities.
    if args.test:
        iiwa_velocities_log = iiwa_velocities.FindLog(simulator.get_context())
        for time, qdot in zip(iiwa_velocities_log.sample_times(),
                              iiwa_velocities_log.data().transpose()):
            # TODO(jwnimmer-tri) We should be able to do better than a 40
            # rad/sec limit, but that's the best we can enforce for now.
            if qdot.max() > 0.1:
                print(f"ERROR: large qdot {qdot} at time {time}")
                sys.exit(1)


if __name__ == '__main__':
    main()
