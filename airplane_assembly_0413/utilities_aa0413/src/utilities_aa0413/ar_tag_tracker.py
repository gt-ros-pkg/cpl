#! /usr/bin/python

import rospy
import roslib
roslib.load_manifest('utilities_aa0413')
import tf

from ar_track_alvar.msg import AlvarMarkers

import utilities_aa0413.locations

ROS_TOPIC_BINMARKES  = "ar_pose_marker"
ROS_TOPIC_SIM_UPDATE = "update_ar_pose_marker"

####################################################
# var
####################################################
pub             = None
sub 		= None
latest_msg 	= None
markers         = []
webcam_to_w     = None

####################################################
# functions
####################################################

def init():
    global sub, pub
    sub = rospy.Subscriber(ROS_TOPIC_BINMARKES, AlvarMarkers, the_cb)
    pub = rospy.Publisher(ROS_TOPIC_SIM_UPDATE, AlvarMarkers)  


def the_cb(msg):
    global latest_msg
    latest_msg = msg

    # update markers
    global markers
    markers = msg.markers
    for i in range(len(msg.markers)):

        new_marker = True

        for j in range(len(markers)):
            if markers[j].id == msg.markers[i].id:
                 new_marker = False
                 markers[j] = msg.markers[i]

        if new_marker:
             markers.append(msg.markers[i])
            

def get_latest_msg():
    return latest_msg

def get_markers():
    return markers


def sim_update(msg):
    pub.publish(msg)

def get_bin_pose(binid):
   
    if get_latest_msg() is None:
         return None

    for marker in get_latest_msg().markers:
         if marker.id == binid:
              return marker.pose.pose

    return None

def locationname2binid(lname):

    (l, o) = utilities_aa0413.locations.get_location_by_name(lname)

    for marker in get_latest_msg().markers:
         pose = marker.pose.pose
         if  abs(l[0] - pose.position.x) + abs(l[1] - pose.position.y) + abs(l[2] - pose.position.z) < 0.1:
              return marker.id

    return None
 
    

def binid2locationname(binid):

    pose = get_bin_pose(binid)

    lnames = utilities_aa0413.locations.get_location_names()

    for lname in lnames:

       (l, o) = utilities_aa0413.locations.get_location_by_name(lname)
   
       #print(abs(l[0] - pose.position.x) + abs(l[1] - pose.position.y) + abs(l[2] - pose.position.z) )
       if  abs(l[0] - pose.position.x) + abs(l[1] - pose.position.y) + abs(l[2] - pose.position.z) < 0.1: 
           return lname

    return None

def is_bin_in_workspace(binid):
     lname = binid2locationname(binid)
     return utilities_aa0413.locations.is_workspace_location(lname)

def get_empty_workspace_location():

    lnames = utilities_aa0413.locations.get_location_names()

    for lname in lnames:

          if utilities_aa0413.locations.is_workspace_location(lname) and locationname2binid(lname) is None:
              return lname

    return None


def get_empty_nonworkspace_location():

    lnames = utilities_aa0413.locations.get_location_names()

    for lname in lnames:

          if not utilities_aa0413.locations.is_workspace_location(lname) and locationname2binid(lname) is None:
              return lname

    return None



def sim_movebinin(binid):
   lname = get_empty_workspace_location()

   if lname is None:
      return False

   move_bin_to_location(binid, lname)
   return True

def sim_movebinout(binid):
   lname = get_empty_nonworkspace_location()

   if lname is None:
      return False

   move_bin_to_location(binid, lname)
   return True

def move_bin(binid, l, o):
   msg    = get_latest_msg()
   for i in range(len(msg.markers)):
      if msg.markers[i].id == binid:

           msg.markers[i].pose.pose.position.x = l[0]
           msg.markers[i].pose.pose.position.y = l[1]
           msg.markers[i].pose.pose.position.z = l[2] 

           msg.markers[i].pose.pose.orientation.x = o[0] 
           msg.markers[i].pose.pose.orientation.y = o[1] 
           msg.markers[i].pose.pose.orientation.z = o[2] 
           msg.markers[i].pose.pose.orientation.w = o[3] 

   sim_update(msg)

def move_bin_to_location(binid, lname):
   (l, o) = utilities_aa0413.locations.get_location_by_name(lname)
   move_bin(binid, l, o)

####################################################
# test
####################################################

def main():
    rospy.init_node('feaouihtf9438wenf')
    init()

    while (get_latest_msg() is None and not rospy.is_shutdown()):
        rospy.sleep(1)
        print('waiting')

    print(get_latest_msg())


if __name__ == '__main__' :
    main()























































































