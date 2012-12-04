#ifndef UNIVERSAL_HARDWARE_H
#define UNIVERSAL_HARDWARE_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <pr2_hardware_interface/hardware_interface.h>
#include <ros/ros.h>

#define RT_PERIODS_PER_STATE 8

namespace ur10_controller_manager 
{

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

    pr2_hardware_interface::HardwareInterface *hw_;

  private:
    ros::NodeHandle node_;
    bool is_halted_;
    bool halt_motors_;
    int rt_periods_passed_;

    std::vector<ur10_controller_manager::UniversalJoint*> actuators;
    ros::Time last_published_;
};

}

#endif // UNIVERSAL_HARDWARE_H
