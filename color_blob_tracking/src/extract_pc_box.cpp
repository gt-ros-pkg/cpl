#include <stdio.h>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

//#include <opencv2/core/core.hpp>
//#include <opencv2/nonfree/features2d.hpp>
//#include <opencv2/legacy/legacy.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/crop_box.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
//#include <Eigen/SVD>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include <color_blob_tracking/hsl_rgb_conversions.h>

//using namespace cv;
using namespace std;

typedef pcl::PointXYZRGB PRGB;
typedef pcl::PointCloud<PRGB> PCRGB;

void transPoints(const PCRGB &pc_in, const Eigen::Matrix4f &trans, PCRGB &pc_out);

void transPoints(const PCRGB &pc_in, const Eigen::Matrix4f &trans, PCRGB &pc_out)
{
    Eigen::MatrixXf m(4,pc_in.size());
    for(size_t i=0;i<pc_in.size();i++) {
        m(0,i) = pc_in[i].x; m(1,i) = pc_in[i].y; m(2,i) = pc_in[i].z; m(3,i) = 1;
    }
    m = trans * m;
    for(size_t i=0;i<pc_in.size();i++) {
        PRGB pt;
        pt.x = m(0,i); pt.y = m(1,i); pt.z = m(2,i); pt.rgb = pc_in[i].rgb;
        pc_out.push_back(pt);
    }
}

namespace color_blob_tracking
{
class ExtractPCBox
{
  private:
    ros::NodeHandle nh;
    geometry_msgs::PoseStamped cube_pose;
    Eigen::Vector4f box_size;
    ros::Subscriber pc_sub;
    ros::Publisher pc_pub;
    tf::TransformListener tf_list;
  public:
    ExtractPCBox(geometry_msgs::PoseStamped cube_pose_, Eigen::Vector4f box_size_) :
      nh(""), cube_pose(cube_pose_), box_size(box_size_)
    {
      pc_sub = nh.subscribe("camera", 1, &ExtractPCBox::pcCallback, this);
      pc_pub = nh.advertise<sensor_msgs::PointCloud2>("pc_out", 1);
    }
    void pcCallback(const PCRGB::ConstPtr pc_in);
      
};

void ExtractPCBox::pcCallback(const PCRGB::ConstPtr pc_in)
{
  //cv::Mat 
	//cv::imshow("Image Display", cv_ptr->image);
  //printf("header frameid %s\n", pc_in->header.frame_id.c_str());
  pcl::CropBox<PRGB> crop_box;
  //crop_box.setMin(Eigen::Vector4f(-5.0, -5.0, 0.0, 1.0));
  //crop_box.setMax(Eigen::Vector4f(5.0, 5.0, 2.0, 1.0));
  crop_box.setInputCloud(pc_in);
  //vector<int> inds;
  //crop_box.filter(inds);
  geometry_msgs::PoseStamped box_pose;
  tf_list.waitForTransform(pc_in->header.frame_id, cube_pose.header.frame_id, 
                           ros::Time(0), ros::Duration(10.0));
  box_pose.header.stamp = ros::Time(0);
  try {
    tf_list.transformPose(pc_in->header.frame_id, cube_pose, box_pose);
  }
  catch (tf::TransformException ex) {
    ROS_ERROR("%s", ex.what());
  }
  Eigen::Affine3d box_pose_eig;
  tf::Pose box_pose_tf;
  tf::poseMsgToTF(box_pose.pose, box_pose_tf);
  tf::TransformTFToEigen(box_pose_tf, box_pose_eig);
  crop_box.setTransform(box_pose_eig.inverse().cast<float>());

  crop_box.setMin(Eigen::Vector4f(0.0, 0.0, 0.0, 1.0));
  crop_box.setMax(box_size);

  PCRGB extracted_pc;
  crop_box.filter(extracted_pc);
  /*
  for(uint32_t u=0;u<pc_in->width;u++)
    for(uint32_t v=0;v<pc_in->height;v++) {
      if(u*pc_in->width+v)
        extracted_pc.push_back(pc_in->at(u, v));
    }
    */
  extracted_pc.header = pc_in->header;
  pc_pub.publish(extracted_pc);
}


}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "extract_pc_box");
  geometry_msgs::PoseStamped cube_pose;
  cube_pose.pose.position.x = atof(argv[1]);
  cube_pose.pose.position.y = atof(argv[2]);
  cube_pose.pose.position.z = atof(argv[3]);
  cube_pose.pose.orientation.x = atof(argv[4]);
  cube_pose.pose.orientation.y = atof(argv[5]);
  cube_pose.pose.orientation.z = atof(argv[6]);
  cube_pose.pose.orientation.w = atof(argv[7]);
  cube_pose.header.frame_id = "/base_link";
  Eigen::Vector4f box_size(atof(argv[8]), atof(argv[9]), atof(argv[10]), 1.0);
  color_blob_tracking::ExtractPCBox ex_pc_cube(cube_pose, box_size);
  ros::spin();
  return 0;
}
