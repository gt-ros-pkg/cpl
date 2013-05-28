#! /usr/bin/python

import rospy
import roslib
roslib.load_manifest('utilities_aa0413')
import tf

from ar_track_alvar.msg import AlvarMarkers

import utilities_aa0413.locations


####################################################
# var
####################################################


####################################################
# functions
####################################################

def init():
    global sub


def lookup_transform_block(frame1, frame2):

    lr = tf.TransformListener();

    while (rospy.is_shutdown() == False):
        try:
            (trans, rot) = lr.lookupTransform(frame1, frame2, rospy.Time(0))
            T  = lr.fromTranslationRotation(trans, rot)
            print 'Lookup transform succeeded'
            return T

        except:
            print 'Lookup transform fail'
            rospy.sleep(1)

####################################################
# test
####################################################

def main():
    rospy.init_node('feaouihtf9438wenf')
    

if __name__ == '__main__' :
    main()























































































