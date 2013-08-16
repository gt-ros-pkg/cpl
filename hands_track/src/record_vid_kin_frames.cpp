#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_broadcaster.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/median.hpp>

#include <sstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cv.h>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace boost::accumulators;

typedef pcl::PointXYZRGB PRGB;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

class handsTracker{

private:
  VideoWriter outputVideo;

  //declare frame globally
  ros::Time last_img_time;
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc;
  ros::NodeHandle nh;
  
  //kinect frame subscribe
  image_transport::Subscriber kinect_rgb_sub;
  //Point cloud subscribe	
  ros::Subscriber kinect_pc_sub ;
  
  //Mats to be used
  Mat img_of_interest, bin_img, ioi_hsv;
  Mat temp_h, temp_s, temp_v, label_img;

  cv_bridge::CvImagePtr cv_ptr;
  geometry_msgs::Quaternion init_quaternion;
  string frame_of_reference; 

public:
  handsTracker();
  ~handsTracker();
  void kin_rgb_fram_sub(const sensor_msgs::ImageConstPtr& msg);
  void frame_process();
  void kin_pc_sub(sensor_msgs::PointCloud2::ConstPtr pc_msg) ;
};
//constructor
handsTracker::handsTracker ()
{
  
  int img_ht=480, img_wd=640;
  double green_mean =  80.33/180, blue_mean =  126.993/180;
  double hard_th_green= 6.0, hard_th_blue= 2.0;
  int frames_to_hold =1;
    
  img_of_interest.create(img_ht, img_wd, CV_8UC3);
  bin_img.create(img_ht, img_wd, CV_8UC1);
  label_img.create(img_ht, img_wd, CV_32FC1);
  temp_h.create(img_ht, img_wd, CV_8UC1);
  temp_s.create(img_ht, img_wd, CV_8UC1);
  temp_v.create(img_ht, img_wd, CV_8UC1);
  frame_of_reference = "/kinect0_rgb_optical_frame";
    
  double dub_zer=0.0, dub_one=1.0;
  init_quaternion.x=dub_zer; init_quaternion.y=dub_zer; init_quaternion.z=dub_zer; init_quaternion.z=dub_one;
    
  image_transport::ImageTransport hand_tracking(nh);
  //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
  //cur_pc = temp_pc;
  kinect_rgb_sub = hand_tracking.subscribe
    ("kinect0/rgb/image_rect_color", frames_to_hold, &handsTracker::kin_rgb_fram_sub, this);
    
  outputVideo.open("kinect_stuff.avi",CV_FOURCC('P','I','M','1'),30.0, img_of_interest.size());
  if(!outputVideo.isOpened()){cout<<"Cant Open Video";exit(-1);}
}

void handsTracker::frame_process()
{

   
  if(!img_of_interest.empty())
    {
      outputVideo << img_of_interest;
	
    }

}

//callback - grabs frame from kinect and processes it
void handsTracker::kin_rgb_fram_sub(const sensor_msgs::ImageConstPtr& msg)
{
    
  last_img_time = msg->header.stamp;
  try
    {
      cv_ptr = cv_bridge::toCvCopy(msg,  sensor_msgs::image_encodings::BGR8);
    }
  catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    
  //    namedWindow("show",1);

  img_of_interest = cv_ptr->image;
    
  //debug
  //cout << "\nNo.of channels"<< img_of_interest.channels();
    
  //debug
  //imshow("show", glob_img_of_interest);
  //waitKey(0);
    
  //process
  frame_process();
}

//callback - grabs point cloud
void handsTracker::kin_pc_sub(sensor_msgs::PointCloud2::ConstPtr pc_msg) 
{
  pcl::fromROSMsg(*pc_msg, *cur_pc);
    
}

handsTracker::~handsTracker()
{
  outputVideo.release();
}


//MAIN
int main(int argc, char **argv)
{

  
  ros::init(argc, argv, "hand_tracker"); 
  handsTracker begin_track;

  
  ros::spin();

      

  return 0;
}

