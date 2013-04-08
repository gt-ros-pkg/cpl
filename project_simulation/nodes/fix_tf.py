#!/usr/bin/env python  

import roslib
roslib.load_manifest('project_simulation')

import rospy
import tf

PUB_RATE = 60


if __name__ == '__main__':
    rospy.init_node('tf_broadcaster')
    br_cam_kin = tf.TransformBroadcaster()
    rate = rospy.Rate(PUB_RATE)
    while not rospy.is_shutdown():
        '''br_cam_kin.sendTransform((0.46789905508028007, -0.4088075565973479, 
                                  -0.3455194750182502),
                                 (-0.27615693462127333, 0.026578535053463227, 
                                   0.002035468187515766, 0.9607428302122536),
                         rospy.Time.now(),
                         "/kinect0_rgb_optical_frame",
                         "/lifecam1_optical_frame")'''

        br_cam_kin.sendTransform((0.260675, 0.037183, 1.903672),
                                 (0.5437498073554586, 0.8197467095728095, -0.14192994971578896, -0.11048696085569207),
                         rospy.Time.now(),
                         "/lifecam1_optical_frame", 
                         "/base_link")
        
        
        br_cam_kin.sendTransform((0.433613, -0.406085, 1.375427),
                                 (0.486451129927709, 0.7524142009645931, -0.3769641006844861, -0.23481106271639962),
                         rospy.Time.now(),
                         "/kinect0_rgb_optical_frame", 
                         "/base_link")
        rate.sleep()
