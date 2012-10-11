import roslib; roslib.load_manifest('visual_servo')
import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, TwistStamped
import visual_servo.msg

def init_client():
  # separate action clients for each arm
  self.vs_action_l_client = actionlib.SimpleActionClient('l_vs_controller/vsaction', visual_servo.msg.VisualServoAction)
  self.vs_action_l_client.wait_for_server()
  self.vs_action_r_client = actionlib.SimpleActionClient('r_vs_controller/vsaction', visual_servo.msg.VisualServoAction)
  self.vs_action_r_client.wait_for_server()

def your_function(which_arm = 'l'):
  start_pose = PoseStamped()
  goal = visual_servo.msg.VisualServoGoal(pose = start_pose)
  if which_arm == 'l':
    vs_client = self.vs_action_l_client
  else:
    vs_client = self.vs_action_r_client

  vs_client.send_goal(goal)
  vs_client.wait_for_result()

  return vs_client.get_result()

if __name__ == '__main__':
  try:
    rospy.init_node('example')
    init_client()
    result = your_function()
    print "Result:", ', '.join([str(n) for n in result.sequence])
  except rospy.ROSInterruptException:
    print "error"
