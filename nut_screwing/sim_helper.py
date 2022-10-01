import logging
import os
import time

import numpy as np

from pydrake.geometry import Meshcat
from pydrake.multibody.tree import JointIndex

def StartMeshcat(open_window=False):
    logging.getLogger('drake').setLevel(logging.WARNING)
    meshcat = Meshcat()
    web_url = meshcat.web_url()

    print(f'Meshcat is now available at {web_url}')
    os.system(f'xdg-open {web_url}')
    return meshcat


class MeshcatJointSlidersThatPublish():

    def __init__(self,
                 meshcat,
                 plant,
                 publishing_system,
                 root_context,
                 lower_limit=-10.,
                 upper_limit=10.,
                 resolution=0.01):
        """
        Creates an meshcat slider for each joint in the plant.  Unlike the
        JointSliders System, we do not expect this to be used in a Simulator.
        It simply updates the context and calls Publish directly from the
        slider callback.

        Args:
            meshcat:      A Meshcat instance.

            plant:        A MultibodyPlant. publishing_system: The System whose
                          Publish method will be called.  Can be the entire
                          Diagram, but can also be a subsystem.

            publishing_system:  The system to call publish on.  Probably a
                          MeshcatVisualizerCpp.

            root_context: A mutable root Context of the Diagram containing both
                          the ``plant`` and the ``publishing_system``; we will
                          extract the subcontext's using `GetMyContextFromRoot`.

            lower_limit:  A scalar or vector of length robot.num_positions().
                          The lower limit of the slider will be the maximum
                          value of this number and any limit specified in the
                          Joint.

            upper_limit:  A scalar or vector of length robot.num_positions().
                          The upper limit of the slider will be the minimum
                          value of this number and any limit specified in the
                          Joint.

            resolution:   A scalar or vector of length robot.num_positions()
                          that specifies the step argument of the FloatSlider.

        Note: Some publishers (like MeshcatVisualizer) use an initialization
        event to "load" the geometry.  You should call that *before* calling
        this method (e.g. with `meshcat.load()`).
        """

        def _broadcast(x, num):
            x = np.array(x)
            assert len(x.shape) <= 1
            return np.array(x) * np.ones(num)

        lower_limit = _broadcast(lower_limit, plant.num_positions())
        upper_limit = _broadcast(upper_limit, plant.num_positions())
        resolution = _broadcast(resolution, plant.num_positions())

        self._meshcat = meshcat
        self._plant = plant
        self._plant_context = plant.GetMyContextFromRoot(root_context)
        self._publishing_system = publishing_system
        self._publishing_context = publishing_system.GetMyContextFromRoot(
            root_context)

        self._sliders = []
        positions = plant.GetPositions(self._plant_context)
        slider_num = 0
        for i in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(i))
            low = joint.position_lower_limits()
            upp = joint.position_upper_limits()
            for j in range(joint.num_positions()):
                index = joint.position_start() + j
                description = joint.name()
                if joint.num_positions() > 1:
                    description += f"[{j}]"
                meshcat.AddSlider(value=positions[index],
                                  min=max(low[j], lower_limit[slider_num]),
                                  max=min(upp[j], upper_limit[slider_num]),
                                  step=resolution[slider_num],
                                  name=description)
                self._sliders.append(description)
                slider_num += 1

    def Publish(self):
        old_positions = self._plant.GetPositions(self._plant_context)
        positions = np.zeros((len(self._sliders), 1))
        for i, s in enumerate(self._sliders):
            positions[i] = self._meshcat.GetSliderValue(s)
        if not np.array_equal(positions, old_positions):
            self._plant.SetPositions(self._plant_context, positions)
            self._publishing_system.Publish(self._publishing_context)
            return True
        return False

    def Run(self, callback=None):
        print("Press the 'Stop JointSliders' button in Meshcat to continue.")
        self._meshcat.AddButton("Stop JointSliders")
        while self._meshcat.GetButtonClicks("Stop JointSliders") < 1:
            if self.Publish() and callback:
                callback(self._plant_context)
            time.sleep(.1)

        self._meshcat.DeleteButton("Stop JointSliders")
