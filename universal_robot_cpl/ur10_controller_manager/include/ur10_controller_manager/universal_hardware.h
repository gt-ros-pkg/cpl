#ifndef UNIVERSAL_HARDWARE_H
#define UNIVERSAL_HARDWARE_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <pr2_hardware_interface/hardware_interface.h>
#include <ros/ros.h>

class UniversalHardware
{
  public:
    UniversalHardware(const std::string& name);
    
    ~UniversalHardware();

    void update(bool reset, bool halt);

    void init();

    pr2_hardware_interface::HardwareInterface *hw_;

  private:
    ros::NodeHandle node_;
    bool halt_motors_;

    std::vector<pr2_hardware_interface::Actuator*> actuators;
    ros::Time last_published_;
};

#endif // UNIVERSAL_HARDWARE_H
