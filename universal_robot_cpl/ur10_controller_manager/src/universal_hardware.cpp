#include <ur10_controller_manager/universal_hardware.h>

#include "ur10_ctrl_iface/robotinterface.h"
#include "ur10_ctrl_iface/Configuration.h"
#include "ur10_ctrl_iface/microprocessor_commands.h"
#include "ur10_ctrl_iface/microprocessor_definitions.h"

using namespace ur10_controller_manager;

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

UniversalJoint::UniversalJoint(std::string s) : pr2_hardware_interface::Actuator(s)
{
}

UniversalJoint::~UniversalJoint()
{
}

UniversalHardware::UniversalHardware(const std::string& name) :
  hw_(0), 
  //node_(ros::NodeHandle(name)), 
  is_halted_(true),
  halt_motors_(true),
  rt_periods_passed_(0),
  loop_iter(0)
{
  printf("Loading robot configuration...");
  if(!configuration_load()) {
    printf("\nLoading configuration failed (.urcontrol dir should be in ~/).\n");
    exit(EXIT_FAILURE);
  }
  printf("success!");

  for(int i=0;i<6;i++) {
    actuator_names[i] = ACTUATOR_NAMES[i];
    q_des[i] = 0.0;
    qd_des[i] = 0.0;
    qdd_des[i] = 0.0;
  }
}

UniversalHardware::~UniversalHardware()
{
  delete hw_;
}

void UniversalHardware::updateState(bool reset, bool halt)
{
  static uint32_t i;
  //rt_periods_passed_ = (rt_periods_passed_+1) % RT_PERIODS_PER_STATE;
  if(true || !rt_periods_passed_) {
    // We should expect a new state to be available from the controller
    robotinterface_read_state_blocking(); // this shouldn't block for longer than 0.001s

    // Record feedback
    robotinterface_get_actual(q_act, qd_act);
    robotinterface_get_actual_current(i_act);
    if(loop_iter == 0)
      for(i = 0; i < 6; i++)
        q_act_first[i] = q_act[i];

    if(robotinterface_get_robot_mode() == ROBOT_RUNNING_MODE)
      robotinterface_set_robot_freedrive_mode();
    //robotinterface_command_empty_command();
    robotinterface_command_position_velocity_acceleration(q_act_first, zero_vector, zero_vector);
    robotinterface_send();

    for(i = 0; i < 6; i++) {
      actuators[i]->state_.position_ = q_act[i];
      actuators[i]->state_.velocity_ = qd_act[i];
      actuators[i]->state_.last_measured_current_ = i_act[i];
    }


    /* TODO IMPLEMENT THESE FEEDBACKS
    void robotinterface_get_actual_accelerometers(double *ax, double *ay,
        double *az);
    double robotinterface_get_tcp_force_scalar();
    void robotinterface_get_tcp_force(double *F);
    void robotinterface_get_tcp_speed(double *V);
    double robotinterface_get_tcp_power();
    double robotinterface_get_power();
    */
  }

  if(halt)
    is_halted_ = true;
  if(reset)
    is_halted_ = false;
  loop_iter++;
}

void UniversalHardware::updateActuators()
{
  static uint32_t i;
  if(!rt_periods_passed_) {
    // Command arm
    if(!is_halted_) {
      for(i = 0; i < 6; i++) {
        q_des[i] = actuators[i]->q_des;
        qd_des[i] = actuators[i]->qd_des;
        qdd_des[i] = actuators[i]->qdd_des;
      }
      robotinterface_command_position_velocity_acceleration(q_des,qd_des,qdd_des);
    } else {
      robotinterface_command_velocity(zero_vector);
    }
    robotinterface_send();
  }
}

void UniversalHardware::init()
{
  hw_ = new pr2_hardware_interface::HardwareInterface();
  hw_->current_time_ = ros::Time::now();
  last_published_ = hw_->current_time_;
  for(uint32_t i = 0; i < 6; i++) {
    ur10_controller_manager::UniversalJoint* actuator = 
      new ur10_controller_manager::UniversalJoint(actuator_names[i]);
    hw_->addActuator((pr2_hardware_interface::Actuator*) actuator);
    actuators.push_back(actuator);
  }
}

void UniversalHardware::startRobot()
{

  // UR10 Interface setup

#if 0
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
#endif

  printf("Opening robot interface...");
  int error = open_interface(100);
  if(error == -1) {
    printf("\nrobotinterface_open() failed\n");
    exit(EXIT_FAILURE);
  } else if(error == -2) {
    printf("\nRobot is still not connected.\n");
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

void UniversalHardware::initializeJoints(double joint_delta)
{
  printf("Initializing robot\n");
  if(initialize_joints(joint_delta)) {
      robotinterface_close();
      printf("Unable to initialize robot\n");
      exit(EXIT_FAILURE);
  }
  printf("Robot initialized\n\n\n\n");
}

void stall_robot(int cycles)
{
  while(cycles-- > 0) {
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
    //if(!ros::ok())
    //  return -2;
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
    //if(!ros::ok())
    //  return -2;
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
      //if(!ros::ok())
      //  return -2;
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
