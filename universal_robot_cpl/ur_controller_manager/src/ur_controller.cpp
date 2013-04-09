#include <stdio.h>

#include <ros/ros.h>
#include <realtime_tools/realtime_publisher.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Wrench.h>
#include <std_msgs/Float64.h>
#include "ur_ctrl_iface/robotinterface.h"
#include "ur_ctrl_iface/Configuration.h"
#include "ur_ctrl_iface/microprocessor_commands.h"
#include "ur_ctrl_iface/microprocessor_definitions.h"
#include "ur_controller_manager/URJointStates.h"
#include "ur_controller_manager/URModeStates.h"
#include "ur_controller_manager/URJointCommand.h"
#include "ur_controller_manager/URModeCommand.h"
#include <math.h>

using namespace std;

#define CMD_TIMEOUT 3
#define CONTROL_RATE 125
#define MSG_BUFFER_SIZE 100

const char* ACTUATOR_NAMES[6] = { 
  "shoulder_pan_joint",
  "shoulder_lift_joint",
  "elbow_joint",
  "wrist_1_joint",
  "wrist_2_joint",
  "wrist_3_joint"};

const char* ROBOT_MODES[11] = {
  "ROBOT_RUNNING_MODE",
  "ROBOT_FREEDRIVE_MODE",
  "ROBOT_READY_MODE",
  "ROBOT_INITIALIZING_MODE",
  "ROBOT_SECURITY_STOPPED_MODE",
  "ROBOT_EMERGENCY_STOPPED_MODE",
  "ROBOT_FATAL_ERROR_MODE",
  "ROBOT_NO_POWER_MODE",
  "ROBOT_NOT_CONNECTED_MODE",
  "ROBOT_SHUTDOWN_MODE",
  "ROBOT_SAFEGUARD_STOP_MODE"};

const char* JOINT_MODES[19] = {
  "JOINT_PART_D_CALIBRATION_MODE",
  "JOINT_BACKDRIVE_MODE",
  "JOINT_POWER_OFF_MODE",
  "JOINT_EMERGENCY_STOPPED_MODE",
  "JOINT_CALVAL_INITIALIZATION_MODE",
  "JOINT_ERROR_MODE",
  "JOINT_FREEDRIVE_MODE",
  "JOINT_SIMULATED_MODE 244 /* Probably not used besides inside P",
  "JOINT_NOT_RESPONDING_MODE",
  "JOINT_MOTOR_INITIALISATION_MODE",
  "JOINT_BOOTING_MODE",
  "JOINT_PART_D_CALIBRATION_ERROR_MODE",
  "JOINT_BOOTLOADER_MODE",
  "JOINT_CALIBRATION_MODE",
  "JOINT_SECURITY_STOPPED_MODE",
  "JOINT_FAULT_MODE",
  "JOINT_RUNNING_MODE",
  "JOINT_INITIALISATION_MODE",
  "JOINT_IDLE_MODE"};

namespace ur_controller_manager
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
int initialize_joints(double* delta_move);

const double zero_vector[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

class URController
{
  private:
    ros::NodeHandle* nh;
    realtime_tools::RealtimePublisher<sensor_msgs::JointState> js_pub;
    realtime_tools::RealtimePublisher<ur_controller_manager::URJointStates> state_pub;
    realtime_tools::RealtimePublisher<ur_controller_manager::URModeStates> mode_pub;
    ros::Subscriber cmd_joint_sub;
    ros::Subscriber cmd_mode_sub;
    ros::Subscriber set_tcp_sub;
    ros::Subscriber set_tcp_payload_sub;
    ros::Subscriber set_tcp_wrench_sub;
    int64_t loop_counter;
    int64_t latest_cmd_loop;
    bool was_emergency_stopped;
    bool change_tcp;
    bool change_tcp_payload;
    bool change_tcp_wrench;

    /////////////////////////// Robot Joint States ////////////////////////////
    double q_act[6];
    double qd_act[6];
    double i_act[6];
    double acc_x[6];
    double acc_y[6];
    double acc_z[6];
    double tcp_force_scalar;
    double tcp_force[6];
    double tcp_speed[6];
    double tcp_power;
    double power;
    double q_des[6];
    double qd_des[6];
    double qdd_des[6];
    double i_des[6];
    double moment_des[6];
    double tcp_wrench[6];
    double tcp_pose[6];
    double tcp_payload;
    ///////////////////////////////////////////////////////////////////////////

    //////////////////////////// Robot Mode States ////////////////////////////
    uint8_t robot_mode_id;
    bool is_power_on_robot;
    bool is_security_stopped;
    bool is_emergency_stopped;
    bool is_extra_button_pressed; /* The button on the back side of the screen */
    bool is_power_button_pressed;  /* The big green button on the controller box */
    bool is_safety_signal_such_that_we_should_stop; /* This is from the safety stop interface */
    uint8_t joint_mode_ids[6];
    ///////////////////////////////////////////////////////////////////////////

    ////////////////////////// Robot Error Messages ///////////////////////////
    struct message_t msg_buffer[MSG_BUFFER_SIZE];
    char msg_text_buffers[MSG_BUFFER_SIZE][100];
    int msg_count;
    ///////////////////////////////////////////////////////////////////////////

    /////////////////////////// Robot Joint Commands //////////////////////////
    uint8_t cmd_mode;
    double q_cmd[6];
    double qd_cmd[6];
    double qdd_cmd[6];
    ///////////////////////////////////////////////////////////////////////////

    /////////////////////////// Robot Mode Commands ///////////////////////////
    bool set_robot_ready_mode;
    bool set_robot_running_mode;
    bool set_robot_freedrive_mode;
    bool unlock_security_stop;
    bool security_stop;
    char joint_code;
    int error_state;
    int error_argument;
    double set_tcp[6];
    double set_tcp_payload;
    double set_tcp_wrench[6];
    ///////////////////////////////////////////////////////////////////////////

    double last_q_cmd[6];

  public:
    URController(ros::NodeHandle* _nh);
    void getRobotStates();
    void pubRobotStates();
    void getRobotModeStates();
    void pubRobotModeStates();
    void commandJoints();
    void commandModes();
    void resetModes();
    void initializeJoints(double* joint_delta);
    void cmdJointCallback(const URJointCommand::ConstPtr& cmd);
    void cmdModeCallback(const URModeCommand::ConstPtr& cmd);
    void setTCPCallback(const geometry_msgs::Twist::ConstPtr& tcp_msg);
    void setTCPPayloadCallback(const std_msgs::Float64::ConstPtr& tcp_payload_msg);
    void setTCPWrench(const geometry_msgs::Wrench::ConstPtr& tcp_wrench_msg);
    void startRobot();
    void controlLoop();
};

URController::URController(ros::NodeHandle* _nh) :
  nh(_nh),
  js_pub(*_nh, "/joint_states", 1),
  state_pub(*_nh, "/ur_joint_states", 1),
  mode_pub(*_nh, "/ur_mode_states", 1),
  loop_counter(0),
  latest_cmd_loop(-10000),
  was_emergency_stopped(false),
  change_tcp(false),
  change_tcp_payload(false),
  change_tcp_wrench(false)
{
  js_pub.msg_.name.resize(6);
  js_pub.msg_.position.resize(6,-9999.0);
  js_pub.msg_.velocity.resize(6,-9999.0);
  js_pub.msg_.effort.resize(6,-9999.0);
  for(int i=0;i<MSG_BUFFER_SIZE;i++)
    msg_buffer[i].text = msg_text_buffers[i];

  cmd_joint_sub = _nh->subscribe<URJointCommand>("/ur_joint_command", 1, 
                                           &URController::cmdJointCallback, this);
  cmd_mode_sub = _nh->subscribe<URModeCommand>("/ur_mode_command", 10, 
                                           &URController::cmdModeCallback, this);
  set_tcp_sub = _nh->subscribe<geometry_msgs::Twist>("/ur_set_tcp", 1, 
                                           &URController::setTCPCallback, this);
  set_tcp_payload_sub = _nh->subscribe<std_msgs::Float64>("/ur_set_tcp_payload", 1, 
                                           &URController::setTCPPayloadCallback, this);
  set_tcp_wrench_sub = _nh->subscribe<geometry_msgs::Wrench>("/ur_set_tcp_wrench", 1, 
                                           &URController::setTCPWrench, this);
  cmd_mode = 20;

  printf("Loading robot configuration...");
  if(!configuration_load()) {
    printf("\nLoading configuration failed (.urcontrol dir should be in ~/).\n");
    exit(EXIT_FAILURE);
  }
  printf("success!");
}

void URController::getRobotStates()
{
  static int i;

  // joint states
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
  robotinterface_get_tcp_wrench(tcp_wrench);
  robotinterface_get_tcp(tcp_pose);
  tcp_payload = robotinterface_get_tcp_payload();

  // mode states
  robot_mode_id = robotinterface_get_robot_mode();
  is_power_on_robot = robotinterface_is_power_on_robot();
  is_security_stopped = robotinterface_is_security_stopped();
  is_emergency_stopped = robotinterface_is_emergency_stopped();
  is_extra_button_pressed = robotinterface_is_extra_button_pressed(); 
  /* The button on the back side of the screen */
  is_power_button_pressed = robotinterface_is_power_button_pressed();  
  /* The big green button on the controller box */
  is_safety_signal_such_that_we_should_stop = 
    robotinterface_is_safety_signal_such_that_we_should_stop(); 
  /* This is from the safety stop interface */
  for(i=0;i<6;i++)
    joint_mode_ids[i] = robotinterface_get_joint_mode(i);

  // read error codes
  msg_count = robotinterface_get_message_count();
  for(i=0;i<msg_count;i++) {
    robotinterface_get_message(&msg_buffer[i]);
  }
}

void URController::pubRobotStates()
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
      state_pub.msg_.tcp_wrench[i] = tcp_wrench[i];
      state_pub.msg_.tcp_pose[i] = tcp_pose[i];
      state_pub.msg_.acc_x[i] = acc_x[i];
      state_pub.msg_.acc_y[i] = acc_y[i];
      state_pub.msg_.acc_z[i] = acc_z[i];
      state_pub.msg_.moment_des[i] = moment_des[i];
    }
    state_pub.msg_.tcp_force_scalar = tcp_force_scalar;
    state_pub.msg_.tcp_payload = tcp_payload;
    state_pub.msg_.power = power;

    state_pub.msg_.header.stamp = now;
    state_pub.msg_.header.seq = loop_counter;
    state_pub.unlockAndPublish();
  }

  if(mode_pub.trylock()) {
    mode_pub.msg_.robot_mode = ROBOT_MODES[robot_mode_id];
    mode_pub.msg_.robot_mode_id = robot_mode_id;
    mode_pub.msg_.is_power_on_robot = is_power_on_robot;
    mode_pub.msg_.is_security_stopped = is_security_stopped;
    mode_pub.msg_.is_emergency_stopped = is_emergency_stopped;
    mode_pub.msg_.is_extra_button_pressed = is_extra_button_pressed;
    mode_pub.msg_.is_power_button_pressed = is_power_button_pressed;
    mode_pub.msg_.is_safety_signal_such_that_we_should_stop = is_safety_signal_such_that_we_should_stop;
    for(i=0;i<6;i++) {
      mode_pub.msg_.joint_mode_ids[i] = joint_mode_ids[i];
      mode_pub.msg_.joint_modes[i] = JOINT_MODES[joint_mode_ids[i]-JOINT_MODE_BAR];
    }
    mode_pub.msg_.messages.resize(msg_count);
    for(i=0;i<msg_count;i++) {
      mode_pub.msg_.messages[i].timestamp = msg_buffer[i].timestamp;
      mode_pub.msg_.messages[i].source = msg_buffer[i].source;
      mode_pub.msg_.messages[i].text = msg_buffer[i].text;
    }
    mode_pub.msg_.header.stamp = now;
    mode_pub.msg_.header.seq = loop_counter;
    mode_pub.unlockAndPublish();
  }
}

void URController::initializeJoints(double* joint_delta)
{
  printf("Initializing robot\n");
  if(initialize_joints(joint_delta)) {
      robotinterface_close();
      printf("Unable to initialize robot\n");
      exit(EXIT_FAILURE);
  }
  /*
  robotinterface_read_state_blocking();
  robotinterface_get_target(last_q_cmd, qd_des, qdd_des);
  robotinterface_command_position_velocity_acceleration(last_q_cmd, qd_des, qdd_des);
  robotinterface_send();
  */
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

void URController::commandJoints()
{
  if(robot_mode_id != ROBOT_RUNNING_MODE) {
    robotinterface_command_empty_command();
    //robotinterface_command_position_velocity_acceleration(last_q_cmd, zero_vector, zero_vector);
  }
  if(loop_counter - latest_cmd_loop >= CMD_TIMEOUT) {
    robotinterface_command_velocity(zero_vector);
    //robotinterface_command_position_velocity_acceleration(last_q_cmd, zero_vector, zero_vector);
#if 0
    if(loop_counter % CONTROL_RATE*10 == 0)
      printf("No command, loop_counter: %d, latest_cmd_loop: %d\n", loop_counter, latest_cmd_loop);
#endif
    return;
  }

  if(cmd_mode == URJointCommand::CMD_EMPTY)
    robotinterface_command_empty_command();
  else if(cmd_mode == URJointCommand::CMD_VELOCITY)
    robotinterface_command_velocity(qd_cmd);
  else if(cmd_mode == URJointCommand::CMD_POS_VEL_ACC) {
    robotinterface_command_position_velocity_acceleration(q_cmd, qd_cmd, qdd_cmd);
    for(int i=0;i<6;i++) last_q_cmd[i] = q_cmd[i];
  }
  else {
    //robotinterface_command_position_velocity_acceleration(q_des, zero_vector, zero_vector);
    robotinterface_command_empty_command();
    if(loop_counter % CONTROL_RATE*1 == 0)
      printf("Bad cmd_mode: %d\n", cmd_mode);
  }
}

void URController::commandModes()
{
  if(set_robot_ready_mode)
    robotinterface_set_robot_ready_mode();
  if(set_robot_running_mode)
    robotinterface_set_robot_running_mode();
  if(set_robot_freedrive_mode)
    robotinterface_set_robot_freedrive_mode();
  if(unlock_security_stop)
    robotinterface_unlock_security_stop();
  if(security_stop)
    robotinterface_security_stop(joint_code, error_state, error_argument);
  if(change_tcp) {
    change_tcp = false;
    robotinterface_set_tcp(set_tcp);
  }
  if(change_tcp_payload) {
    change_tcp_payload = false;
    robotinterface_set_tcp_payload(set_tcp_payload);
  }
  if(change_tcp_wrench) {
    change_tcp_wrench = false;
    robotinterface_set_tcp_wrench(set_tcp_wrench, true);
  }
}

void URController::resetModes()
{
  set_robot_ready_mode = false;
  set_robot_running_mode = false;
  set_robot_freedrive_mode = false;
  unlock_security_stop = false;
  security_stop = false;
}

void URController::cmdJointCallback(const URJointCommand::ConstPtr& cmd)
{
  static int i;
  cmd_mode = cmd->mode;
  for(i=0;i<6;i++) {
    q_cmd[i] = cmd->q_des[i];
    qd_cmd[i] = cmd->qd_des[i];
    qdd_cmd[i] = cmd->qdd_des[i];
  }
  latest_cmd_loop = loop_counter;
}

void URController::cmdModeCallback(const URModeCommand::ConstPtr& cmd)
{
  set_robot_ready_mode = set_robot_ready_mode || cmd->robot_ready_mode;
  set_robot_running_mode = set_robot_running_mode || cmd->robot_running_mode;
  set_robot_freedrive_mode = set_robot_freedrive_mode || cmd->robot_freedrive_mode;
  unlock_security_stop = unlock_security_stop || cmd->unlock_security_stop;
  security_stop = security_stop || cmd->security_stop;
  if(cmd->security_stop) {
    joint_code = cmd->joint_code;
    error_state = cmd->error_state;
    error_argument = cmd->error_argument;
  }
}

void URController::setTCPCallback(const geometry_msgs::Twist::ConstPtr& tcp_msg)
{
  change_tcp = true;
  set_tcp[0] = tcp_msg->linear.x;
  set_tcp[1] = tcp_msg->linear.y;
  set_tcp[2] = tcp_msg->linear.z;
  set_tcp[3] = tcp_msg->angular.x;
  set_tcp[4] = tcp_msg->angular.y;
  set_tcp[5] = tcp_msg->angular.z;
}

void URController::setTCPPayloadCallback(const std_msgs::Float64::ConstPtr& tcp_payload_msg)
{
  change_tcp_payload = true;
  set_tcp_payload = tcp_payload_msg->data;
}

void URController::setTCPWrench(const geometry_msgs::Wrench::ConstPtr& tcp_wrench_msg)
{
  change_tcp_wrench = true;
  set_tcp_wrench[0] = tcp_wrench_msg->force.x;
  set_tcp_wrench[1] = tcp_wrench_msg->force.y;
  set_tcp_wrench[2] = tcp_wrench_msg->force.z;
  set_tcp_wrench[3] = tcp_wrench_msg->torque.x;
  set_tcp_wrench[4] = tcp_wrench_msg->torque.y;
  set_tcp_wrench[5] = tcp_wrench_msg->torque.z;
}

void URController::controlLoop()
{
  while(ros::ok()) {
    // If we have retriggered the emergency stop, reboot it
    if(is_emergency_stopped)
      was_emergency_stopped = true;
    else
      if(was_emergency_stopped && robot_mode_id == ROBOT_INITIALIZING_MODE) {
        robotinterface_power_on_robot();
        was_emergency_stopped = false;
        latest_cmd_loop = -10000;
      }

    robotinterface_read_state_blocking();
    getRobotStates();
    pubRobotStates();
    resetModes();
    ros::spinOnce();
    //robotinterface_set_tcp(zero_vector);
    //robotinterface_set_tcp_payload(3.0);
    commandJoints();
    commandModes();
    robotinterface_send();
    loop_counter++;
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

int initialize_joints(double* delta_move)
{
  int i=0,j;
  double speed_vector[6];
  do {
    if(!ros::ok())
      return -3;
    robotinterface_read_state_blocking();
    for (j=0; j<6; ++j) {
      //speed_vector[j] = 0.25 * sin(((double)i) / 80.0);
      speed_vector[j] = (robotinterface_get_joint_mode(j) == 
          JOINT_INITIALISATION_MODE || robotinterface_get_joint_mode(j) ==
          JOINT_IDLE_MODE) ? delta_move[j] : 0.0;
    }
    i++;
    robotinterface_command_velocity(speed_vector);
    robotinterface_send();
  } while(robotinterface_get_robot_mode() == ROBOT_INITIALIZING_MODE);
  return robotinterface_get_robot_mode() != ROBOT_RUNNING_MODE;
}

}

using namespace ur_controller_manager;

int main(int argc, char* argv[])
{
  string name = "ur_controller";
  double joint_delta[6];
  if(argc == 2) {
    for(int j=0;j<6;j++)
      joint_delta[j] = atof(argv[1]);
  }
  else {
    if(argc == 7) {
      for(int j=0;j<6;j++)
        joint_delta[j] = atof(argv[1+j]);
    }
    else {
      printf("Need joint deltas\n");
      return -1;
    }
  }

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

