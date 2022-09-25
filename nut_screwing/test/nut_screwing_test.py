import numpy as np

import nut_screwing.sim_helper as sh
import nut_screwing.differential_controller as diff_c
import nut_screwing.open_loop_controller as ol_c

import nut_screwing.run_manipulator as run

import unittest

from pydrake.math import RigidTransform, RotationMatrix

class TestNutLib(unittest.TestCase):
    def test_make_gripper_frames(self):
        print('test_make_gripper_frames')
        X_G = {"initial": RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2.0), [0, -0.25, 0.25])}
        X_O = {"initial": RigidTransform(RotationMatrix.MakeZRotation(np.pi/2.0), [-.4, .0, 0.]),
               "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi),[-.4, -.6, 0.])}
        X_G_dbg, times_dbg = run.make_gripper_frames(X_G, X_O)
        assert X_G_dbg is not None
        assert times_dbg is not None
    

if __name__ == "__main__":
    unittest.main()
