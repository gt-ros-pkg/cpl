#include <ur10_controller_manager/universal_hardware.h>

UniversalHardware::UniversalHardware(const std::string& name) :
  hw_(0), node_(ros::NodeHandle(name)), halt_motors_(true)
{
}

UniversalHardware::~UniversalHardware()
{
  delete hw_;
}

void UniversalHardware::update(bool reset, bool halt)
{
  for(uint32_t i = 0; i < 6; i++) {
    actuators[i]->state_.position_ = 3;
    actuators[i]->state_.velocity_ = 4;
    actuators[i]->state_.last_measured_effort_ = 5;
  }
}

const char* actuator_names[6] = { "shoulder_pan_motor",
                                  "shoulder_lift_motor",
                                  "elbow_motor",
                                  "wrist_1_motor",
                                  "wrist_2_motor",
                                  "wrist_3_motor"};

void UniversalHardware::init()
{
  hw_ = new pr2_hardware_interface::HardwareInterface();
  hw_->current_time_ = ros::Time::now();
  last_published_ = hw_->current_time_;
  for(uint32_t i = 0; i < 6; i++) {
    pr2_hardware_interface::Actuator* actuator = 
      new pr2_hardware_interface::Actuator(actuator_names[i]);
    hw_->addActuator(actuator);
    actuators.push_back(actuator);
  }
}
