#ifndef UNIVERSAL_HARDWARE_H
#define UNIVERSAL_HARDWARE_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <pr2_hardware_interface/hardware_interface.h>
#include <ros/ros.h>

#define RT_PERIODS_PER_STATE 1

namespace ur10_controller_manager 
{

const char* ACTUATOR_NAMES[6] = { "shoulder_pan_joint",
                                  "shoulder_lift_joint",
                                  "elbow_joint",
                                  "wrist_1_joint",
                                  "wrist_2_joint",
                                  "wrist_3_joint"};

class UniversalJoint : public pr2_hardware_interface::Actuator
{
  public:
    UniversalJoint(std::string s);
    ~UniversalJoint();
    double q_des;
    double qd_des;
    double qdd_des;
};

class UniversalHardware
{
  public:
    UniversalHardware(const std::string& name);
    
    ~UniversalHardware();

    void updateState(bool reset, bool halt);
    void updateActuators();

    void init();
    void startRobot();
    void initializeJoints(double delta);

    pr2_hardware_interface::HardwareInterface *hw_;

  private:
    //ros::NodeHandle node_;
    bool is_halted_;
    bool halt_motors_;
    int rt_periods_passed_;
    uint64_t loop_iter;

  public:
    std::vector<ur10_controller_manager::UniversalJoint*> actuators;
    ros::Time last_published_;
    double q_act[6];
    double q_act_first[6];
    double qd_act[6];
    double i_act[6];
    double q_des[6];
    double qd_des[6];
    double qdd_des[6];

    const char* actuator_names[6];
};

}

#endif // UNIVERSAL_HARDWARE_H
