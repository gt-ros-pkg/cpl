import numpy as np
import sys
import roslib
roslib.load_manifest("ur_cart_move")
roslib.load_manifest("hrl_geom")
from ur_cart_move.ur_cart_move import load_ur_robot
from geometry_msgs.msg import PoseStamped
from hrl_geom.pose_converter import PoseConv
import rospy
from sensor_msgs.msg import JointState
from ur_analytical_ik import inverse_kin, UR10_A, UR10_D, UR10_L

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
def make_joint_state(q=[0.]*6, qd=[0.]*6, effort=[0.]*6):
    js = JointState()
    js.header.stamp = rospy.Time.now()
    for i, joint_name in enumerate(JOINT_NAMES):
        js.name.append(joint_name)
        js.position.append(q[i])
        js.velocity.append(qd[i])
        js.effort.append(effort[i])
    return js

def best_sol(sols, q_guess, weights):
    valid_sols = []
    for sol in sols:
        test_sol = np.ones(6)*9999.
        for i in range(6):
            for add_ang in [-2.*np.pi, 0, 2.*np.pi]:
                test_ang = sol[i] + add_ang
                if (abs(test_ang) <= 2.*np.pi and 
                    abs(test_ang - q_guess[i]) < abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol)
    if len(valid_sols) == 0:
        return None
    best_sol_ind = np.argmin(np.sum((weights*(valid_sols - np.array(q_guess)))**2,1))
    return valid_sols[best_sol_ind]

rospy.init_node("ipython")
arm1, kin, arm_behav = load_ur_robot(topic_prefix="/sim1")
arm2, kin, arm_behav = load_ur_robot(topic_prefix="/sim2")
#pub = rospy.Publisher('test', PoseStamped)
np.set_printoptions(precision=3)
qs, diffs = [], []
for i1 in range(4):
    for i2 in range(4):
        print i1, i2
        for i3 in range(4):
            for i4 in range(4):
                for i5 in range(4):
                    for i6 in range(4):
                        q = np.array([i1*np.pi/2., i2*np.pi/2., i3*np.pi/2., 
                                      i4*np.pi/2., i5*np.pi/2., i6*np.pi/2.])
                        q = (np.random.rand(6)-.5)*4*np.pi
                        x = kin.forward(q)
                        #qsol = kin.inverse(x, q)
                        sols = inverse_kin(x, UR10_A, UR10_D, UR10_L, q[5])
                        qsol = best_sol(sols, q, [1.]*6)
                        if qsol is None:
                            qsol = [999.]*6
                        diff = np.sum(np.abs(np.array(qsol) - q))
                        if rospy.is_shutdown():
                            sys.exit()
                        if diff > 0.05:
                            arm1.cmd_pos_vel_acc(q,[0.]*6,[0.]*6)
                            arm2.cmd_pos_vel_acc(qsol,[0.]*6,[0.]*6)
                            #pub.publish(PoseConv.to_pose_stamped_msg('/base_link', kin
                            print np.array(sols)
                            print 'Best q:', qsol
                            print 'Actual:', np.array(q)
                            print 'Diff:  ', q - qsol
                            print 'Difdiv:', (q - qsol)/np.pi
                            print i1-3, i2-3, i3-3, i4-3, i5-3, i6-3
                            raw_input()
