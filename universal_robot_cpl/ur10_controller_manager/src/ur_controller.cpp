#include <stdio.h>

#include <ros/ros.h>
#include <realtime_tools/realtime_publisher.h>
#include <sensor_msgs/JointState.h>
#include "ur10_ctrl_iface/robotinterface.h"
#include "ur10_ctrl_iface/Configuration.h"
#include "ur10_ctrl_iface/microprocessor_commands.h"
#include "ur10_ctrl_iface/microprocessor_definitions.h"
#include "ur10_controller_manager/URState.h"
#include <math.h>

using namespace std;

const char* ACTUATOR_NAMES[6] = { "shoulder_pan_joint",
                                  "shoulder_lift_joint",
                                  "elbow_joint",
                                  "wrist_1_joint",
                                  "wrist_2_joint",
                                  "wrist_3_joint"};

namespace ur10_controller_manager
{

// Busy wait which keeps the robot active.
void stall_robot(int cycles);
// returns 0 if successful
int open_interface(int retries);
// returns 0 if successful
int wait_robot_mode(int mode, int retries);
// returns 0 if successful
int powerup_robot(int max_attempts, int retries);
// returns 0 if successful
int initialize_joints(double delta_move);

const double zero_vector[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

class URController
{
  private:
    ros::NodeHandle* nh;
    realtime_tools::RealtimePublisher<sensor_msgs::JointState> js_pub;
    realtime_tools::RealtimePublisher<ur10_controller_manager::URState> state_pub;
    uint32_t loop_counter;

    /////////////////////////// Robot Joint States ////////////////////////////
    double q_act[6];
    double qd_act[6];
    double i_act[6];
    double acc_x[20];
    double acc_y[20];
    double acc_z[20];
    double tcp_force_scalar;
    double tcp_force[6];
    double tcp_speed[6];
    double tcp_power;
    double power;
    double q_des[6];
    double qd_des[6];
    double qdd_des[6];
    double i_des[6];
    double moment_des[20];
    ///////////////////////////////////////////////////////////////////////////

  public:
    URController(ros::NodeHandle* _nh);
    void getRobotJointStates();
    void pubRobotJointStates();
    void initializeJoints(double joint_delta);
    void startRobot();
    void controlLoop();
};

URController::URController(ros::NodeHandle* _nh) :
  nh(_nh),
  js_pub(*_nh, "/joint_states", 1),
  state_pub(*_nh, "/ur_joint_states", 1),
  loop_counter(0)
{
  js_pub.msg_.name.resize(6);
  js_pub.msg_.position.resize(6,-9999.0);
  js_pub.msg_.velocity.resize(6,-9999.0);
  js_pub.msg_.effort.resize(6,-9999.0);

  printf("Loading robot configuration...");
  if(!configuration_load()) {
    printf("\nLoading configuration failed (.urcontrol dir should be in ~/).\n");
    exit(EXIT_FAILURE);
  }
  printf("success!");
}

void URController::getRobotJointStates()
{
  robotinterface_get_actual(q_act, qd_act);
  robotinterface_get_actual_current(i_act);
  robotinterface_get_actual_accelerometers(acc_x, acc_y, acc_z);
  tcp_force_scalar = robotinterface_get_tcp_force_scalar();
  robotinterface_get_tcp_force(tcp_force);
  robotinterface_get_tcp_speed(tcp_speed);
  tcp_power = robotinterface_get_tcp_power();
  power = robotinterface_get_power();
  robotinterface_get_target(q_des, qd_des, qdd_des);
  robotinterface_get_target_current(i_des);
  robotinterface_get_target_moment(moment_des);
}

void URController::pubRobotJointStates()
{
  static int i;
  ros::Time now = ros::Time::now();
  if(js_pub.trylock()) {
    for(i=0;i<6;i++) {
      js_pub.msg_.name[i] = ACTUATOR_NAMES[i];
      js_pub.msg_.position[i] = q_act[i];
      js_pub.msg_.velocity[i] = qd_act[i];
      js_pub.msg_.effort[i] = i_act[i];
    }
    js_pub.msg_.header.stamp = now;
    js_pub.msg_.header.seq = loop_counter;
    js_pub.unlockAndPublish();
  }

  if(state_pub.trylock()) {
    for(i=0;i<6;i++) {
      state_pub.msg_.acutator_names[i] = ACTUATOR_NAMES[i];
      state_pub.msg_.q_act[i] = q_act[i];
      state_pub.msg_.qd_act[i] = qd_act[i];
      state_pub.msg_.i_act[i] = i_act[i];
      state_pub.msg_.tcp_force[i] = tcp_force[i];
      state_pub.msg_.tcp_speed[i] = tcp_speed[i];
      state_pub.msg_.q_des[i] = q_des[i];
      state_pub.msg_.qd_des[i] = qd_des[i];
      state_pub.msg_.qdd_des[i] = qdd_des[i];
      state_pub.msg_.i_des[i] = i_des[i];
    }
    for(i=0;i<20;i++) {
      state_pub.msg_.acc_x[i] = acc_x[i];
      state_pub.msg_.acc_y[i] = acc_y[i];
      state_pub.msg_.acc_z[i] = acc_z[i];
      state_pub.msg_.moment_des[i] = moment_des[i];
    }
    state_pub.msg_.tcp_force_scalar = tcp_force_scalar;
    state_pub.msg_.power = power;

    state_pub.msg_.header.stamp = now;
    state_pub.msg_.header.seq = loop_counter;
    state_pub.unlockAndPublish();
  }
}

void URController::initializeJoints(double joint_delta)
{
  printf("Initializing robot\n");
  if(initialize_joints(joint_delta)) {
      robotinterface_close();
      printf("Unable to initialize robot\n");
      exit(EXIT_FAILURE);
  }
  printf("Robot initialized\n\n\n\n");
}

void URController::startRobot()
{

  // UR10 Interface setup
  printf("Opening robot interface...");
  int error = open_interface(100);
  if(error == -1) {
    printf("\nrobotinterface_open() failed\n");
    exit(EXIT_FAILURE);
  } else if(error == -2) {
    printf("\nRobot is still not connected.\n");
    exit(EXIT_FAILURE);
  } else if(error == -3) {
    printf("\nNode terminated.\n");
    exit(EXIT_FAILURE);
  }
  printf("success!\n");
  

  printf("Wating for ROBOT_NO_POWER_MODE...");
  if(wait_robot_mode(ROBOT_NO_POWER_MODE, 10000)) {
    printf("Robot never went into this mode\n");
    exit(EXIT_FAILURE);
  }
  stall_robot(400);
  printf("success!\n");

  printf("Powering up robot...\n");
  if(powerup_robot(15,400)) {
    robotinterface_close();
    printf("Unable to power up robot\n");
    exit(EXIT_FAILURE);
  }
  printf("success!\n");
}

void URController::controlLoop()
{
  while(ros::ok()) {
    robotinterface_read_state_blocking();
    getRobotJointStates();
    pubRobotJointStates();
    robotinterface_command_velocity(zero_vector);
    robotinterface_send();
  }
}


void stall_robot(int cycles)
{
  while(cycles-- > 0) {
    if(!ros::ok())
      return;
    robotinterface_read_state_blocking();
    robotinterface_command_velocity(zero_vector);
    robotinterface_send();
  }
}

int open_interface(int retries)
{
  if(!robotinterface_open(0))
    return -1;
  while(retries-- > 0 && !robotinterface_is_robot_connected()) {
    if(!ros::ok())
      return -3;
    robotinterface_read_state_blocking();
    robotinterface_command_velocity(zero_vector);
    robotinterface_send();
  }
  if(robotinterface_is_robot_connected())
    return 0;
  return -2;
}

int wait_robot_mode(int mode, int retries)
{
  while(retries-- > 0) {
    if(robotinterface_get_robot_mode() == mode)
      break;
    if(!ros::ok())
      return -3;
    robotinterface_read_state_blocking();
    robotinterface_command_velocity(zero_vector);
    robotinterface_send();
  }
  return robotinterface_get_robot_mode() != mode;
}

int powerup_robot(int max_attempts, int retries)
{
  while(max_attempts-- > 0) {
    robotinterface_power_on_robot();
    while(retries-- > 0) {
      if(!ros::ok())
        return -3;
      robotinterface_read_state_blocking();
      robotinterface_command_velocity(zero_vector);
      robotinterface_send();
      if(robotinterface_is_power_on_robot()) 
        break;
    }
  }
  if(robotinterface_is_power_on_robot())
    return 0;
  return -1;
}

int initialize_joints(double delta_move)
{
  int i=0,j;
  double speed_vector[6];
  do {
    robotinterface_read_state_blocking();
    for (j=0; j<6; ++j) {
      //speed_vector[j] = 0.25 * sin(((double)i) / 80.0);
      speed_vector[j] = (robotinterface_get_joint_mode(j) == 
          JOINT_INITIALISATION_MODE) ? delta_move : 0.0;
    }
    i++;
    robotinterface_command_velocity(speed_vector);
    robotinterface_send();
  } while(robotinterface_get_robot_mode() == ROBOT_INITIALIZING_MODE);
  return robotinterface_get_robot_mode() != ROBOT_RUNNING_MODE;
}

}

using namespace ur10_controller_manager;

int main(int argc, char* argv[])
{
  string name = "ur_controller";
  if(argc < 2) {
    printf("Need joint delta\n");
    return -1;
  }
  double joint_delta = atof(argv[1]);

  // Make this thread RT priority
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
  ////////////////////////////

  ros::init(argc, argv, name);
  ros::NodeHandle nh(name);

  URController urc(&nh);
  urc.startRobot();
  urc.initializeJoints(joint_delta);
  urc.controlLoop();
  return 0;
}

