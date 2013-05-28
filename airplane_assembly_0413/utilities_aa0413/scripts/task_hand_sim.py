#! /usr/bin/python

import rospy
import roslib
roslib.load_manifest('utilities_aa0413')
import tf

from ar_track_alvar.msg import *

from geometry_msgs.msg import *

import numpy
import random

import utilities_aa0413.g
import utilities_aa0413.locations
import utilities_aa0413.task_description
import utilities_aa0413.ar_tag_tracker

import hrl_geom.pose_converter

ROS_TOPIC_LEFTHAND  = "left_hand"
ROS_TOPIC_RIGHTHAND = "right_hand"

REF_FRAME 	= 'base_link'
WEBCAM_FRAME 	= 'lifecam1_optical_frame'

PUB_RATE 	= 10

FPS 		= 30

MOVING_TIME 	= 1

REACHABLE_RADIUS = 0.6

####################################################
# 
####################################################




####################################################
# main
####################################################


def main():

    print 'running ....'

    rospy.init_node('Task_hand_sim')
    utilities_aa0413.ar_tag_tracker.init()

    lefthandpub  = rospy.Publisher(ROS_TOPIC_LEFTHAND, PoseStamped)
    righthandpub = rospy.Publisher(ROS_TOPIC_RIGHTHAND, PoseStamped)

    lefthand_msg  = PoseStamped()
    righthand_msg = PoseStamped()

    lefthand_msg.header.frame_id  = REF_FRAME
    righthand_msg.header.frame_id = REF_FRAME


    # generate actions
    t       = utilities_aa0413.task_description.taskfile2dict('model.xml')
    actions = utilities_aa0413.task_description.gen_random_task(t)

    print '------------------------'
    print 'Generated actions:'
    print '------------------------'
    begin_time   = rospy.Time.now() + rospy.Duration.from_sec(5)
    current_time = begin_time
    for i in range(len(actions)):
        actions[i]['timing'] = current_time
        current_time         = current_time + rospy.Duration.from_sec(actions[i]['dur'] / FPS)
        print actions[i]['name'] + '     ' + str(actions[i]['dur'] / FPS)

    print 'Total'
    print (current_time - begin_time).to_sec()


    # some stuffs
    rest_location = utilities_aa0413.locations.get_hand_rest_location()
    webcam_to_w   = utilities_aa0413.g.lookup_transform_block(REF_FRAME, WEBCAM_FRAME)

    hand_touch_in_binframe = t['detection']['onedetector']['learnt']['mean']
    print hand_touch_in_binframe

    # do do do
    print '------------------------'
    print 'Start now'
    print '------------------------'
    r              = rospy.Rate(PUB_RATE)
    current_action = -1
    wait_time      = rospy.Duration.from_sec(0)
    while not rospy.is_shutdown():
        
        # print the action
        if current_action+1 < len(actions) and (rospy.Time.now() - actions[current_action+1]['timing'] - wait_time).to_sec() > 0:
            if current_action + 1 >= len(actions) :
                break;
            current_action = current_action + 1
            print actions[current_action]['name'] + '     ' + str(actions[current_action]['dur'] / FPS)


        # calculate hand position
        lefthand_msg.pose.position.x 	= rest_location[0]
        lefthand_msg.pose.position.y 	= rest_location[1] - 0.2
        lefthand_msg.pose.position.z 	= rest_location[2]
        righthand_msg.pose.position.x 	= rest_location[0] 
        righthand_msg.pose.position.y 	= rest_location[1] - 0.1
        righthand_msg.pose.position.z 	= rest_location[2]

        for a in actions:
            # time distance
            d = abs( (rospy.Time.now() - a['timing'] - wait_time).to_sec() )

            # bin location
            if utilities_aa0413.ar_tag_tracker.get_bin_pose(a['bin_id']) is None:

                # space distance
                sd = float("inf")

            else:
                # bin location
                b = utilities_aa0413.ar_tag_tracker.get_bin_pose(a['bin_id']).position
                o = utilities_aa0413.ar_tag_tracker.get_bin_pose(a['bin_id']).orientation
                # convert to hand touch location
                #x = tf.listener.fromTranslationRotation(b, [o.x, o.y, o.z, o.w])
                x = hrl_geom.pose_converter.PoseConv.to_homo_mat(utilities_aa0413.ar_tag_tracker.get_bin_pose(a['bin_id']))
                #print x


                #print 'b1'
                #print webcam_to_w
                b = [b.x, b.y, b.z, 1]
                #print b
                b = webcam_to_w.dot(b)
                #print b

                #print 'b2'
                #print numpy.array(webcam_to_w)
                b2 = x.dot([0, 0, 0, 1])
                b2 = x.dot([float(hand_touch_in_binframe[0][0]), float(hand_touch_in_binframe[1][0]), float(hand_touch_in_binframe[2][0]), 1])
                #print b2
                b2 = numpy.matrix(webcam_to_w).dot(numpy.transpose(b2))
                #print b2
                b2 = [float(b2[0][0]), float(b2[1][0]), float(b2[2][0]), 1]
                #print b2

                b = b2;

                # space distance
                sd = numpy.linalg.norm(numpy.array(rest_location) - b[0:3])
                #sd = numpy.linalg.norm(numpy.array(rest_location) - numpy.transpose(b[:,0:3]))


            # moving hand
            if d < MOVING_TIME and sd > REACHABLE_RADIUS :
                 print 'bin ' + str(a['bin_id']) + ' is too far (distance = ' + str(sd) + ')'
                 wait_time += rospy.Duration.from_sec(1.0 / PUB_RATE)

            elif d < MOVING_TIME:

                 p = d / MOVING_TIME

                 if False : 
                     lefthand_msg.pose.position.x = p * lefthand_msg.pose.position.x + (1-p) * b[0]
                     lefthand_msg.pose.position.y = p * lefthand_msg.pose.position.y + (1-p) * b[1]
                     lefthand_msg.pose.position.z = p * lefthand_msg.pose.position.z + (1-p) * b[2]

                 else :
                     righthand_msg.pose.position.x = p * righthand_msg.pose.position.x + (1-p) * b[0]
                     righthand_msg.pose.position.y = p * righthand_msg.pose.position.y + (1-p) * b[1]
                     righthand_msg.pose.position.z = p * righthand_msg.pose.position.z + (1-p) * b[2]

        # add noise
        lefthand_msg.pose.position.x 	+= random.gauss(0, 0.005)
        lefthand_msg.pose.position.y 	+= random.gauss(0, 0.005)
        lefthand_msg.pose.position.z 	+= random.gauss(0, 0.005)
        righthand_msg.pose.position.x 	+= random.gauss(0, 0.005)
        righthand_msg.pose.position.y 	+= random.gauss(0, 0.005)
        righthand_msg.pose.position.z 	+= random.gauss(0, 0.005)

        # publish
        lefthand_msg.header.stamp       = rospy.Time.now()
        righthand_msg.header.stamp      = rospy.Time.now()
        lefthandpub.publish(lefthand_msg)
        righthandpub.publish(righthand_msg)

        r.sleep()





if __name__ == '__main__' :
    main()























































































