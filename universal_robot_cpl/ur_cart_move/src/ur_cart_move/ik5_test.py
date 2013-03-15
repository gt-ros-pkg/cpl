#!/usr/bin/python

import numpy as np
from collections import defaultdict

import roslib
roslib.load_manifest('ur_cart_move')

import rospy

from ur_cart_move.ur_cart_move import load_ur_robot
from hrl_geom.pose_converter import PoseConv

from openravepy import *

def main():
    np.set_printoptions(precision=4, suppress=True)
    rospy.init_node("ur_cart_move")
    arm, kin, arm_behav = load_ur_robot(desc_filename='$(find ur10_description)/ur10_robot.xml')
    env, robot = kin.env, kin.robot

    if False:
        # generate the ik solver
        ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, iktype=IkParameterization.Type.TranslationDirection5D)
        if not ikmodel.load():
            ikmodel.autogenerate()

        while not rospy.is_shutdown():
            with env:
                while not rospy.is_shutdown():
                    target=ikmodel.manip.GetTransform()[0:3,3]+(random.rand(3)-0.5)
                    direction = random.rand(3)-0.5
                    direction /= linalg.norm(direction)
                    print '1'
                    solutions = ikmodel.manip.FindIKSolutions(IkParameterization(Ray(target,direction),IkParameterization.Type.TranslationDirection5D),IkFilterOptions.CheckEnvCollisions)
                    print '2'
                    if solutions is not None and len(solutions) > 0: # if found, then break
                        break
            h=env.drawlinestrip(array([target,target+0.1*direction]),10)
            for i in random.permutation(len(solutions))[0:min(80,len(solutions))]:
                with env:
                    robot.SetDOFValues(solutions[i],ikmodel.manip.GetArmIndices())
                    env.UpdatePublishedBodies()
                time.sleep(0.2)
            h=None

if __name__ == "__main__":
    main()
