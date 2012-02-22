/*
 *  Copyright (c) 2011, A.M.Howard, S.Williams
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of the <organization> nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * In this example, a timer is used to trigger read events instead of the built-in
 * 'spin()' function. The YAML configuration file associated with Example3 provides
 * additional properties, such as a channel number and home position for each servo.
 * The custom 'init_actuator_' function demonstrates how to read these additional
 * fields from the XMLRPS structure. Also, as a demonstration, the various helper
 * functions are called directly, instead of using the all-in-one 'init()' function.
 * Again, a DummyActuator class is used to simulate the operation of a real R/C
 * Servo motor. This removes the need to have specific hardware to test the basic
 * ActuatorArrayDriver system.
 *
 *  Created on: Nov 27, 2011
 *      Author: Stephen Williams
 */

#ifndef EXAMPLE3_DRIVER_H_
#define EXAMPLE3_DRIVER_H_

#include <actuator_array_driver/actuator_array_driver.h>
#include <actuator_array_example/dummy_actuator.h>

namespace actuator_array_example
{

struct Example3JointProperties : public actuator_array_driver::JointProperties
{
  int channel;
  double home;
};

class Example3Driver : public actuator_array_driver::ActuatorArrayDriver<Example3JointProperties>
{
private:

  // Convenience typedef to a map of JointName-JointProperties
  typedef std::map<std::string, Example3JointProperties> JointMap;

  // Create a timer object to trigger servo reads. The other standard choice
  // is to put the node in a custom spin loop that continually reads the servo
  // status. The function spin() is provided in the actuator_array_driver base
  // class for such a purpose.
  ros::Timer timer_;

  // A container of DummyActuator objects, stored by Channel ID
  std::map<int, DummyActuator> actuators_;

public:
  Example3Driver();
  virtual ~Example3Driver();
  void timerCallback(const ros::TimerEvent& e);

  bool init_actuator_(const std::string& joint_name, Example3JointProperties& joint_properties, XmlRpc::XmlRpcValue& joint_data);
  bool command_();
  bool stop_();
  bool home_();
  bool read_(ros::Time ts = ros::Time::now());
};

}
#endif	// EXAMPLE3_DRIVER_H_