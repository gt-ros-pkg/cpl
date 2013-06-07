#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cv.h>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;

//global var
Mat img_of_interest;

void kinect_rgb_subscriber(sensor_msgs::Image::ConstPtr& kin_rgb_img)
{
  cv_bridge::CvImagePtr cv_ptr;
    try
      {
	cv_ptr = cv_bridge::toCvCopy(kin_rgb_img, sensor_msgs::image_encodings::BGR8);
      }
    catch (cv_bridge::Exception& e)
      {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
      }
    
    img_of_interest = cv_ptr->image;
}


//MAIN
int main(int argc, char **argv)
{
  int num_frames_hold = 10;
  //ros initialize node, topic, subscribe
  ros::init(argc, argv, "hand_tracker");
  ros::NodeHandle hand_tracker_node;  
  ros::Subscriber kinect_rgb = hand_tracker_node.subscribe
    ("kinect0/rgb/image_rect_color", num_frames_hold, kinect_rgb_subscriber);

  //Movie Window
  namedWindow( "Track Hands", 2 );
  int delay = 10;
  char c;

  imshow("Track Hands", img_of_interest);
  
  ros::spin();
  
}
