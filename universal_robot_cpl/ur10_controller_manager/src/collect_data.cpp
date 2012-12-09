
#include <stdio.h>
#include <pthread.h>

#include <ros/ros.h>
#include <realtime_tools/realtime_publisher.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <pr2_mechanism_model/joint.h>
#include <pr2_controller_manager/controller_manager.h>

#include <ur10_controller_manager/universal_hardware.h>

using namespace ur10_controller_manager;
using namespace std;

class CollectData
{
  public:
    CollectData(string _name, ros::NodeHandle* nh, UniversalHardware* ur);
    string name;
    ros::NodeHandle* nh;
    UniversalHardware* ur;
    uint64_t loop_iter;
    pr2_controller_manager::ControllerManager* cm; 
    realtime_tools::RealtimePublisher<sensor_msgs::JointState> js_pub;
    ros::Subscriber cmd_sub;
    double cmd;

    void controlLoop();
    void publishJointState();
    void cmdCallback(const std_msgs::Float64::ConstPtr& msg);
};

CollectData::CollectData(string _name, ros::NodeHandle* _nh, UniversalHardware *_ur) : 
  name(_name), 
  nh(_nh),
  ur(_ur),
  loop_iter(0),
  js_pub(*_nh, "/joint_states", 1),
  cmd(0.0)
{

  js_pub.msg_.name.resize(6);
  js_pub.msg_.position.resize(6);
  js_pub.msg_.velocity.resize(6);
  js_pub.msg_.effort.resize(6);

  cmd_sub = _nh->subscribe<std_msgs::Float64>("/test", 1, &CollectData::cmdCallback, this);
  //cm = new pr2_controller_manager::ControllerManager (ur->hw_, *nh);
  
  /*
   TODO for some reason uncommenting this kills the robot...
   can't put it before or the device won't open!
  if (!cm->initXml(root))
  {
      ROS_FATAL("Could not initialize the controller manager");
      return;
  }
  */

  controlLoop();
}

void CollectData::controlLoop()
{
  while(ros::ok()) {
    //ur->freedriveMode();
    ur->testMode();
    //ur->updateActuators();
    publishJointState();
    ros::spinOnce();
    ur->qd_des[4] = cmd;
    loop_iter++;
  }
}

void CollectData::publishJointState()
{
  if(js_pub.trylock()) {
    /*
    for(size_t i=0;i<ur->actuators.size();i++) {
      ur10_controller_manager::UniversalJoint* joint = ur->actuators[i];
      js_pub.msg_.name[i] = actuator_names[i];
      js_pub.msg_.position[i] = joint->state_.position_;
      js_pub.msg_.velocity[i] = joint->state_.velocity_;
      js_pub.msg_.effort[i] = joint->state_.last_measured_current_;
    }
    js_pub.msg_.header.stamp = ros::Time::now();
    js_pub.unlockAndPublish();
    */
    for(int i=0;i<6;i++) {
      js_pub.msg_.name[i] = ur->actuator_names[i];
      js_pub.msg_.position[i] = ur->q_act[i];
      js_pub.msg_.velocity[i] = ur->qd_act[i];
      js_pub.msg_.effort[i] = ur->i_act[i];
    }
    js_pub.msg_.header.stamp = ros::Time::now();
    js_pub.unlockAndPublish();
  }
}

void CollectData::cmdCallback(const std_msgs::Float64::ConstPtr& msg)
{
  cmd = msg->data;
}

int main(int argc, char* argv[])
{
  string name = "collect_data";
  if(argc < 2) {
    printf("Need joint delta\n");
    return -1;
  }

  struct sched_param sch_param;
  pid_t pid;
  printf("Setting RT priority\n");
  pid = getpid();
  sch_param.sched_priority = 99;
  if (sched_setscheduler(pid, SCHED_FIFO, &sch_param) == 0) {
    printf("- Priority set\n");
  } else {
    printf("- Priority not set, error: %i\n", errno);
    exit(EXIT_FAILURE);
  }
  ros::init(argc, argv, name);
  ros::NodeHandle nh(name);

  UniversalHardware ur(name);
  ur.init();

  ur.startRobot();
  double joint_delta = atof(argv[1]);
  ur.initializeJoints(joint_delta);
  CollectData cd(name, &nh, &ur);
  return 0;
}
