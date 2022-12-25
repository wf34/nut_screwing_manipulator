import os
import csv

from pydrake.systems.framework import EventStatus
from pydrake.multibody.tree import MultibodyForces
from pydrake.multibody.math import SpatialForce

import numpy as np

def get_3_vector(name):
    axes = ['x', 'y', 'z']
    return list(map(lambda x : name + '_' + x, axes))

TIME = 'time'
TRANSLATION = get_3_vector('translation')
QUATERNION = ['quaternion_w'] + get_3_vector('quaternion')
VELOCITY = get_3_vector('velocity_a') + get_3_vector('velocity_l')
ACCELERATION = get_3_vector('acceleration_a') + get_3_vector('acceleration_l')
FORCE = get_3_vector('torque') + get_3_vector('force')
HEADER = [TIME] + TRANSLATION + QUATERNION + VELOCITY + ACCELERATION + FORCE


def augment_datum(datum, keys, values):
    if values is None:
        values = [0] * len(keys)

    assert len(keys) == len(values)
    for k, v in zip(keys, values):
        assert k not in datum
        datum[k] = v


class StateMonitor:
    def __init__(self, path, plant):
        self._plant = plant
        
        if os.path.exists(path):
            assert not os.path.isdir(path)
            if os.path.isfile(path):
                print('\n\n\nclearing', os.path.abspath(path))
                os.remove(path)

        self._file = open(path, 'w')
        self._writer = csv.DictWriter(self._file, delimiter=' ', fieldnames=HEADER)
        self._writer.writeheader()
        self._file.flush()
        self._counter = 0

    def add_data(self, time, pose, velocity, acceleration, force=None):
        self._counter += 1
        if 0 == self._counter % 10:
            # logs every 100'th
            datum = {TIME: time}

            t = pose.translation()
            augment_datum(datum, TRANSLATION, t)

            q = pose.rotation().ToQuaternion().wxyz()
            augment_datum(datum, QUATERNION, q)

            augment_datum(datum, VELOCITY, velocity.get_coeffs())
            augment_datum(datum, ACCELERATION, acceleration.get_coeffs())
            augment_datum(datum, FORCE, force.get_coeffs())

            self._writer.writerow(datum)
            self._file.flush()

    def callback(self, root_context):
        nut = self._plant.GetBodyByName("nut")
        nut_context = self._plant.GetMyContextFromRoot(root_context)
        X_WB = nut.EvalPoseInWorld(nut_context)

        contact_results = self._plant.get_contact_results_output_port().Eval(nut_context)

        multibody_forces = MultibodyForces(plant=self._plant)

        assert 0 == contact_results.num_hydroelastic_contacts()
        for i in range(contact_results.num_point_pair_contacts()):
             contact_pair = contact_results.point_pair_contact_info(i)
             sign = 1. if contact_pair.bodyB_index() == nut.index() else -1.
             f_Bc_W = sign*contact_pair.contact_force()
             F_Bc_W = SpatialForce(tau=np.zeros(3), f=f_Bc_W)
             p_WBo = X_WB.translation()
             p_WC = contact_pair.contact_point()
             p_BoC_W = p_WBo - p_WC
             F_Bo_W = F_Bc_W.Shift(p_BoC_W)
             nut.AddInForceInWorld(nut_context, F_Bo_W, multibody_forces)

        self.add_data(root_context.get_time(),
                      X_WB,
                      nut.EvalSpatialVelocityInWorld(nut_context),
                      nut.EvalSpatialAccelerationInWorld(nut_context),
                      nut.GetForceInWorld(nut_context, multibody_forces))
        return EventStatus.DidNothing()
