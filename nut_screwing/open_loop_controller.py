
import numpy as np

from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.multibody import inverse_kinematics
from pydrake.all import (
    PiecewiseQuaternionSlerp,
    PiecewisePolynomial,
    Solve,
    Quaternion,
    TrajectorySource
)

from pydrake.geometry import (Cylinder, GeometryInstance,
                              MakePhongIllustrationProperties)

import experimental_controller as e_c

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


def InterpolateScrewingPose(timestamp, X_WNinit, X_WNgoal):
    # timestamp is not a timestamp here really, more like a fraction of progress towards the goal
    # with 0 : pose at the grasping place before turning,
    #      1 : the pose when a one turning motion was concluded.
    
    p_GgraspO = [0., 0.07, 0.]
    R_GgraspO = RotationMatrix.Identity()
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()

    X_WGpick = X_WNinit.multiply(X_OGgrasp)
    X_WGgoal = X_WNgoal.multiply(X_OGgrasp)
    
    rpy_init = RollPitchYaw(X_WNinit.rotation()).vector()
    rpy_goal = RollPitchYaw(X_WNgoal.rotation()).vector()

    # print(rpy_init, rpy_goal)
    # print(X_WNinit.translation(), X_WNgoal.translation())
    # assert False, 'unimplemented'
    
    # Start by interpolating the yaw angle of the hinge.
    angle_start = rpy_init[2]
    theta = rpy_init[2] + (rpy_goal[2] - rpy_init[2]) * timestamp

    rpy_at_timestamp = rpy_init[:2].tolist() + [theta]
    R_WG_at_timestamp = RollPitchYaw(*rpy_at_timestamp).ToRotationMatrix()
    
    return RigidTransform(R_WG_at_timestamp, X_WGpick.translation())


## Interpolate Pose for entry.
def make_gripper_orientation_trajectory(initial_pose):
  trajectory = PiecewiseQuaternionSlerp()
  trajectory.Append(0.0, initial_pose.rotation())
  trajectory.Append(2.0, RotationMatrix([
    [0.9999781896285951, -0.004815253553629587, -0.00452035400524933],
    [0.004973109578929019, 0.09858770383606971, 0.9951159393927165],
    [-0.004346084241530561, -0.99511671576029, 0.09860950038520559],
  ]))
  trajectory.Append(4.0, RotationMatrix([
    [0.9999886981399402, -0.003906316892014463, -0.002710033344446008],
    [0.004162243031155186, 0.4438264352059445, 0.896103102966023],
    [-0.002297678249548118, -0.896104255151555, 0.443837678181944],
  ]))
  trajectory.Append(6.0, RotationMatrix([
    [0.9999865159362368, -0.0051636507313275815, -0.0005519572741062068],
    [0.004167478314740657, 0.7345360739004784, 0.6785568423230054],
    [-0.0030983980060258543, -0.6785499928892464, 0.7345476887716685],
  ]))
  trajectory.Append(8.0, RotationMatrix([
    [0.9999667068246075, -0.008021451180165265, 0.0014971851303508966],
    [0.006848274618369303, 0.9247421802305565, 0.3805325232318378],
    [-0.004436933299218892, -0.38050960096087194, 0.9247663257274766],
  ]))
  trajectory.Append(10.0, RotationMatrix([
    [0.999951400812072, -0.00938922128722685, 0.003006416071598764],
    [0.009255942508448979, 0.9990898609143617, 0.04163865207234541],
    [-0.0033946343332335715, -0.04160880125335297, 0.9991282120508878],
  ]))
  trajectory.Append(12.0, RotationMatrix([
    [0.9999551350808552, -0.008701118079059099, 0.003744111323593368],
    [0.008611887070014135, 0.9996932254528362, 0.023222626569769707],
    [-0.003945025541427186, -0.023189340824609696, 0.9997233073433862],
  ]))
  trajectory.Append(14.0, RotationMatrix([
    [0.9999680491944266, -0.006966324596669298, 0.003920575456304882],
    [0.0069394919370628005, 0.9999526879432362, 0.00681654800862841],
    [-0.003967876251873429, -0.006789123412660569, 0.9999690814026895],
  ]))
  trajectory.Append(15.0, RotationMatrix([
    [0.9999721247388583, -0.0063624823006780615, 0.0039075010208442715],
    [0.006352774069663243, 0.9999767160318963, 0.002491918073479834],
    [-0.003923264823352601, -0.0024670251394506796, 0.999989260832379],
  ]))
  return trajectory 


def make_gripper_position_trajectory(initial_pose):
  trajectory = PiecewisePolynomial.FirstOrderHold(
      [0., 2., 4., 6., 8., 10., 12., 14., 15.], 
      np.vstack([[initial_pose.translation()],
                 [0.002601232337067979, -0.46300349590701806, 0.6982997416521892],
                 [0.002889750861476299, -0.49807940795779093, 0.563817523099881],
                 [0.003503064673777145, -0.5305570841096271, 0.39162964355795216],
                 [0.004131105841572578, -0.5377652712098702, 0.2154458597275652],
                 [0.004268469231277225, -0.5013511168955569, 0.08684335454632779],
                 [0.003448235158406986, -0.4532557500915213, 0.06549690094407969],
                 [0.003077075668437291, -0.4044722552100217, 0.09172075836079888],
                 [0.003269001860685704, -0.390343642335943, 0.10299734718060621]]).T)
  return trajectory


def InterpolateEntryPose(timestamp,
                         entry_trajectory_rotation,
                         entry_trajectory_translation):
  return RigidTransform(RotationMatrix(Quaternion(entry_trajectory_rotation.value(timestamp))), 
                        entry_trajectory_translation.value(timestamp))


def InterpolateGripperPose(timestamp, X_G, X_O):
    X_WGinit = X_G['initial']
    X_WNinit = X_O['initial']
    X_WNgoal = X_O['goal']
    entry_trajectory_rotation = make_gripper_orientation_trajectory(X_WGinit)
    entry_trajectory_translation = make_gripper_position_trajectory(X_WGinit)

    if timestamp < 15.0:
       # Duration of entry motion is set to 5 seconds.
        return InterpolateEntryPose(timestamp, entry_trajectory_rotation, entry_trajectory_translation)
    if  timestamp >= 15.0 and timestamp < 16.0:
        # Wait for a second to grip the handle.
        return InterpolateEntryPose(15.0, entry_trajectory_rotation, entry_trajectory_translation)
    else:
        # Duration of the screw motion is set to 5 seconds.
        return InterpolateScrewingPose((timestamp - 16.0) / 21.0, X_WNinit, X_WNgoal)


def get_q_trajectory():
    ts = [0.110, 0.410, 0.710, 1.010, 1.310, 1.610, 1.910, 2.210, 2.510, 2.810, 3.110, 3.410, 3.710, 4.010, 4.310, 4.610, 4.910, 5.210, 5.510, 5.810, 6.110, 6.410, 6.710, 7.010, 7.310, 7.610, 7.910, 8.210, 8.510, 8.810, 9.110, 9.410, 9.710, 10.010, 10.310, 10.610, 10.910, 11.210, 11.510, 11.810, 12.110, 12.410, 12.710, 13.010, 13.310, 13.610, 13.910, 14.210, 14.510, 14.810]
    
    qs = np.vstack( \
    [[ -1.57141, 0.11540, 0.00231, -1.09293, -0.00107, 1.70193, 0.00072 ],
    [ -1.57487, 0.17628, 0.01029, -0.85814, -0.00520, 1.92338, 0.00262 ],
    [ -1.57617, 0.21091, 0.01597, -0.81009, -0.00872, 1.98705, 0.00349 ],
    [ -1.57633, 0.23565, 0.01857, -0.79847, -0.01053, 2.02722, 0.00375 ],
    [ -1.57638, 0.25828, 0.02002, -0.79710, -0.01136, 2.06021, 0.00402 ],
    [ -1.57646, 0.28006, 0.02096, -0.80311, -0.01166, 2.08703, 0.00438 ],
    [ -1.57654, 0.30124, 0.02151, -0.81641, -0.01159, 2.10658, 0.00482 ],
    [ -1.57664, 0.32232, 0.02173, -0.83594, -0.01122, 2.11892, 0.00529 ],
    [ -1.57672, 0.34463, 0.02176, -0.85803, -0.01069, 2.12798, 0.00581 ],
    [ -1.57675, 0.36867, 0.02166, -0.88148, -0.01012, 2.13444, 0.00636 ],
    [ -1.57672, 0.39487, 0.02148, -0.90532, -0.00955, 2.13873, 0.00694 ],
    [ -1.57661, 0.42358, 0.02123, -0.92866, -0.00902, 2.14132, 0.00752 ],
    [ -1.57641, 0.45509, 0.02092, -0.95068, -0.00855, 2.14264, 0.00809 ],
    [ -1.57614, 0.48962, 0.02056, -0.97066, -0.00816, 2.14311, 0.00865 ],
    [ -1.57579, 0.52729, 0.02015, -0.98799, -0.00785, 2.14315, 0.00918 ],
    [ -1.57537, 0.56816, 0.01971, -1.00213, -0.00761, 2.14315, 0.00968 ],
    [ -1.57490, 0.61222, 0.01923, -1.01269, -0.00745, 2.14348, 0.01013 ],
    [ -1.57437, 0.65937, 0.01872, -1.01935, -0.00736, 2.14450, 0.01053 ],
    [ -1.57382, 0.70943, 0.01821, -1.02192, -0.00734, 2.14652, 0.01089 ],
    [ -1.57324, 0.76217, 0.01769, -1.02029, -0.00737, 2.14986, 0.01118 ],
    [ -1.57265, 0.81728, 0.01718, -1.01449, -0.00745, 2.15476, 0.01140 ],
    [ -1.57206, 0.87441, 0.01668, -1.00464, -0.00757, 2.16143, 0.01157 ],
    [ -1.57147, 0.93313, 0.01620, -0.99099, -0.00772, 2.17004, 0.01167 ],
    [ -1.57090, 0.99299, 0.01574, -0.97389, -0.00788, 2.18070, 0.01170 ],
    [ -1.57034, 1.05347, 0.01530, -0.95382, -0.00806, 2.19343, 0.01166 ],
    [ -1.56980, 1.11403, 0.01488, -0.93140, -0.00822, 2.20821, 0.01156 ],
    [ -1.56928, 1.17405, 0.01448, -0.90736, -0.00837, 2.22492, 0.01140 ],
    [ -1.56878, 1.23290, 0.01408, -0.88260, -0.00847, 2.24335, 0.01118 ],
    [ -1.56831, 1.28985, 0.01367, -0.85816, -0.00851, 2.26321, 0.01091 ],
    [ -1.56787, 1.34415, 0.01324, -0.83525, -0.00848, 2.28411, 0.01060 ],
    [ -1.56747, 1.39493, 0.01277, -0.81524, -0.00834, 2.30556, 0.01026 ],
    [ -1.56711, 1.44131, 0.01226, -0.79966, -0.00807, 2.32699, 0.00989 ],
    [ -1.56680, 1.48235, 0.01169, -0.79010, -0.00767, 2.34776, 0.00952 ],
    [ -1.56653, 1.51710, 0.01105, -0.78821, -0.00712, 2.36715, 0.00914 ],
    [ -1.56646, 1.52220, 0.01043, -0.81039, -0.00686, 2.34181, 0.00898 ],
    [ -1.56649, 1.52215, 0.00996, -0.83236, -0.00681, 2.32090, 0.00879 ],
    [ -1.56656, 1.51740, 0.00950, -0.86046, -0.00663, 2.30040, 0.00866 ],
    [ -1.56665, 1.50816, 0.00901, -0.89370, -0.00632, 2.27944, 0.00855 ],
    [ -1.56674, 1.49526, 0.00852, -0.93068, -0.00595, 2.25840, 0.00847 ],
    [ -1.56683, 1.47948, 0.00805, -0.97024, -0.00554, 2.23763, 0.00841 ],
    [ -1.56691, 1.46156, 0.00761, -1.01130, -0.00509, 2.21743, 0.00837 ],
    [ -1.56698, 1.44215, 0.00720, -1.05291, -0.00463, 2.19810, 0.00833 ],
    [ -1.56704, 1.42187, 0.00683, -1.09417, -0.00417, 2.17986, 0.00831 ],
    [ -1.56707, 1.40127, 0.00650, -1.13429, -0.00370, 2.16292, 0.00830 ],
    [ -1.56708, 1.38092, 0.00622, -1.17251, -0.00325, 2.14747, 0.00830 ],
    [ -1.56707, 1.36132, 0.00598, -1.20813, -0.00280, 2.13364, 0.00832 ],
    [ -1.56704, 1.34301, 0.00578, -1.24049, -0.00236, 2.12156, 0.00835 ],
    [ -1.56697, 1.32650, 0.00562, -1.26895, -0.00194, 2.11133, 0.00841 ],
    [ -1.56688, 1.31232, 0.00551, -1.29288, -0.00153, 2.10303, 0.00848 ],
    [ -1.56676, 1.30100, 0.00543, -1.31164, -0.00114, 2.09674, 0.00859 ]
    ]).T
    trajectory = PiecewisePolynomial.FirstOrderHold(ts,qs)
    return trajectory


def AddOrientationConstraint(plant, ik, target_frame, R_TG, bounds):
    """Add orientation constraint to the ik problem. Implements an inequality 
    constraint where the axis-angle difference between f_R(q) and R_WG must be
    within bounds. Can be translated to:
    ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
    """
    ik.AddOrientationConstraint(
        frameAbar=plant.GetFrameByName("body"), R_AbarA=R_TG, 
        frameBbar=target_frame, R_BbarB=RotationMatrix(),
        theta_bound=bounds
    )


def AddPositionConstraint(plant, ik, target_frame, p_BQ, p_NG_lower, p_NG_upper):
    """Add position constraint to the ik problem. Implements an inequality
    constraint where f_p(q) must lie between p_WG_lower and p_WG_upper. Can be
    translated to 
    ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
    """
    ik.AddPositionConstraint(
        frameA=plant.GetFrameByName("body"),
        frameB=target_frame,
        p_BQ=p_BQ,
        p_AQ_lower=p_NG_lower, p_AQ_upper=p_NG_upper)


def create_q_keyframes(timestamps, keyframe_poses, plant):
    """Convert end-effector pose list to joint position list using series of 
    InverseKinematics problems. Note that q is 9-dimensional because the last 2 dimensions 
    contain gripper joints, but these should not matter to the constraints.
    @param: keyframe_poses (python list): keyframe_poses[i] contains keyframe X_WG at index i.
    @return: q_keyframes (python_list): q_knots[i] contains IK solution that will give f(q_knots[i]) \approx pose_lst[i].
    """
    q_keyframes = []
    valid_timestamps = []
    internal_trajectory = get_q_trajectory()

    #cc = CustomCost(plant, station)

    for keyframe_index, (keyframe_timestamp, keyframe_pose) in enumerate(zip(timestamps, keyframe_poses)):
        # closing time
        if 15. < keyframe_timestamp < 17.:
            valid_timestamps.append(keyframe_timestamp)
            q_keyframes.append(q_keyframes[-1])
            continue

        #q_nominal = np.array([0., -1.57, 0.1, 0, -1.2, 0, 1.6, 0, 0, 0])
        q_nominal = np.zeros(10,)
        q_nominal[1:8] = e_c.IIWA_DEFAULT_POSITION

        # q_nominal = np.array([0., 0.,  1.51609774,  0., -0.78808917, 0.,  2.36662405,  0., 0., 0.]) # nominal joint for joint-centering.
        # q_nominal = np.array([0., 0., 0.6, 0., -1.75, 0., 1., 0., 0., 0.])
        #q_nominal = np.array([0., -1.56702176,  1.33784888, 0.00572793, -1.24946957, -0.002234, 2.05829444, 0.00836547, 0., 0.])
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q() # Get variables for MathematicalProgram
        # print(q_variables.shape, q_nominal.shape)
        prog = ik.prog() # Get MathematicalProgram
        print(keyframe_timestamp, keyframe_pose.translation())
        q_target = np.zeros(10,)
        keyframe_internal = internal_trajectory.value(keyframe_timestamp).ravel()
        keyframe_internal[:4] /= np.linalg.norm(keyframe_internal[:4])
        q_target[2:9] = keyframe_internal
                
        if 0 == keyframe_index:
            prog.SetInitialGuess(q_variables, q_nominal)
        else:
            prog.SetInitialGuess(q_variables, q_keyframes[-1])
        
        prog.AddCost(np.square(np.dot(q_variables, q_target))) #q_nominal

        if keyframe_timestamp < 16.:
            p_GG_lower = np.array([-0.1, -0.1, -0.1])
            p_GG_upper = np.array([0.1, 0.1, 0.1])
            distance = min(1. / keyframe_timestamp, 1.5)
            if keyframe_timestamp > 11:
                p_GG_upper[2] = distance
                if keyframe_timestamp > 14:
                    p_GG_upper[2] = 0

            AddPositionConstraint(plant, ik, plant.world_frame(),
                                  keyframe_pose.translation(), p_GG_lower, p_GG_upper)
        
        if 5. < keyframe_timestamp < 16.:
            theta_0 = min(3. / keyframe_timestamp + 0.15, np.pi / 4)
            print('theta_0', theta_0)
            AddOrientationConstraint(plant, ik, plant.world_frame(), keyframe_pose.rotation(), theta_0)
        
        #if keyframe_timestamp > 10:
        #    AddAngleBetweenVectorsConstraint(ik, plant.GetFrameByName("nut"), \
        #                                 [0, 1, 0], [0,1,0], 0.1)

        if 10 < keyframe_timestamp < 15. and False:
            distance = min(1. / keyframe_timestamp + 0.05, 1.5)
            print('distance', distance)
            d_lb = distance * -1.
            d_ub = distance
            p_NG_lower = np.array(2*[d_lb+0.1] + [d_lb])
            p_NG_upper = np.array(2*[d_ub+0.1] + [d_ub])
            AddPositionConstraint(plant, ik, plant.GetFrameByName("nut"),
                                  [0, -0.03, 0], p_NG_lower, p_NG_upper)
            #if keyframe_timestamp > 12:
            #    AddAngleBetweenVectorsConstraint(ik, plant.GetFrameByName("nut"),
            #                                     [0, 1, 0], [0,1,0], 0.3)
        

        #theta_1 = 0.1
        #AddOrientationConstraint(ik, plant.GetFrameByName("nut"), RotationMatrix(), theta_1)
#        p_NG_lower = np.array([-0.2, -0.2, -0.2])
#        p_NG_upper = np.array([0.2, 0.2, 0.2])
#        AddPositionConstraint(ik, plant.GetFrameByName("nut"), [0, 0.2, 0], p_NG_lower, p_NG_upper)

        #x = ChooseBestSolver(prog)
        #print(x.name(), dir(x))
        #solver = MakeSolver(x)
        ##print(dir(_))
        #so = SolverOptions()
        #print(so.GetOptions(x))
        
        result = Solve(prog)
        if not result.is_success():
            print(result.GetInfeasibleConstraintNames(prog), '\n')
            print('no sol for i={} ts={:.1f}'.format(keyframe_index, keyframe_timestamp))
            break
        else:
            print('kf #{} is ok at timestamp {:.1f}'.format(keyframe_index, keyframe_timestamp))
            q_keyframes.append(result.GetSolution(q_variables))
            valid_timestamps.append(keyframe_timestamp)
            if len(result.GetInfeasibleConstraintNames(prog)):
                print(result.GetInfeasibleConstraintNames(prog), '\n')

    return np.array(valid_timestamps), np.array(q_keyframes)


def create_open_loop_controller(builder, plant, scene_graph, X_G, X_O, draw_frames):
    # 1) hardcode pivotal poses
    # 2) pivotal transforms are used to interpolate a few `keyframe_poses`
    # 3) ik is solved over `keyframe_poses`.
    
    timestamps = np.linspace(0, 21, 60)
    print(timestamps)
    keyframe_poses = []
    for timestamp in timestamps:
        keyframe_poses.append(InterpolateGripperPose(timestamp, X_G, X_O))
        triad_name = 'keyframe_pose_{}'.format(timestamp)
        if draw_frames:
            AddTriad(plant.get_source_id(),
                     plant.GetBodyFrameIdOrThrow(plant.world_frame().body().index()),
                     scene_graph, name=triad_name, length=0.1, radius=0.004, opacity=0.5,
                    X_FT=keyframe_poses[-1])
    print('keyframe_poses: ', len(keyframe_poses))
    valid_timestamps, q_keyframes = create_q_keyframes(timestamps, keyframe_poses, plant)
    assert len(valid_timestamps) > 0
    q_trajectory = PiecewisePolynomial.CubicShapePreserving(valid_timestamps, q_keyframes[:, 1:8].T)
    q_trajectory_system = builder.AddSystem(TrajectorySource(q_trajectory))
    

    joint_space_traj = np.hstack((valid_timestamps[:, np.newaxis], q_keyframes[:, 1:8]))

    wsg_timestamps = np.array([0., 15., 16., 21.])
    wsg_keyframes = np.array([0.107, 0.107, 0., 0.]).reshape(1,4)
    wsg_trajectory = PiecewisePolynomial.FirstOrderHold(wsg_timestamps, wsg_keyframes)
    wsg_trajectory_system = builder.AddSystem(TrajectorySource(wsg_trajectory))

    return q_trajectory_system.get_output_port(), \
           wsg_trajectory_system.get_output_port(), \
           (timestamps, keyframe_poses), joint_space_traj
