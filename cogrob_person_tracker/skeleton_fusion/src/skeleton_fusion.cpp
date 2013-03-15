#include <iostream>
#include <vector>
#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <openni_tracker_msgs/jointData.h>
#include <openni_tracker_msgs/skeletonData.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf/transform_listener.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZ Point;
using namespace std;


class skeleton_fusion
{
private:
  string fixed_frame_id;
  
  ros::Time last_timestamp;
  ros::Publisher skeletonData_pub;
  ros::Publisher skeleton_cloud_pub_0;
  ros::Publisher skeleton_cloud_pub_1;
  ros::Subscriber skeletonData_sub;
  openni_tracker_msgs::skeletonData person;
  tf::TransformListener tf_listener;
  
public:
  ros::NodeHandle nh_;

  skeleton_fusion():
    nh_("~")
  {
    string temp="openni_depth_frame";
    skeletonData_sub = nh_.subscribe("/skeleton_data", 100, &skeleton_fusion::msgCallback,this);
    skeletonData_pub=nh_.advertise<openni_tracker_msgs::skeletonData>("/person", 1000);
    skeleton_cloud_pub_0=nh_.advertise<sensor_msgs::PointCloud2>("/skeleton_cloud_0",1);
    skeleton_cloud_pub_1=nh_.advertise<sensor_msgs::PointCloud2>("/skeleton_cloud_1",1);
    nh_.param("fixed_frame_id",fixed_frame_id, temp);
    std::cout<<"fixed_frame_id: "<<fixed_frame_id<<std::endl;
    init_variables();
  }


void publishTransform(int joint, string const& child_frame_id)
{
	// Publish the fusioned data to TF

	static tf::TransformBroadcaster br;
	tf::Transform transform;
	transform.setOrigin(tf::Vector3(person.joints[joint].pose.position.x, person.joints[joint].pose.position.y, person.joints[joint].pose.position.z));

	transform.setRotation(tf::Quaternion(0.0, 0.0, 0.0)); // Add joint info
	/*
	// This is for rotation the data
	tf::Transform change_frame;
	change_frame.setOrigin(tf::Vector3(0, 0, 0)); // If we want to change the offset. 
	tf::Quaternion frame_rotation;
	frame_rotation.setEulerZYX(1.5708, 0, 1.5708); // The rotation This is 90 degrees on Z and X.
    change_frame.setRotation(frame_rotation);

    transform = change_frame * transform;
	*/
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(),"/person_tf", child_frame_id));
}



void publishGlobalSkeleton()
{
  skeletonData_pub.publish(person);
  std::cout<<"publish person"<<std::endl;
}

void publishTF()
{
 	publishTransform(0, "head");
	publishTransform(1, "neck");
	publishTransform(2, "torso");

	publishTransform(6, "left_shoulder");
	publishTransform(7, "left_elbow");
	publishTransform(9, "left_hand");

	publishTransform(12, "right_shoulder");
	publishTransform(13, "right_elbow");
	publishTransform(15, "right_hand");

	publishTransform(17, "left_hip");
	publishTransform(18, "left_knee");
	publishTransform(20, "left_foot");

	publishTransform(21, "right_hip");
	publishTransform(22, "right_knee");
	publishTransform(24, "right_foot");
}

pcl::PointCloud<Point> ExtractPointCloud(const openni_tracker_msgs::skeletonData::ConstPtr &msg)
{
  pcl::PointCloud<Point> pt_cloud;
  pt_cloud.header.frame_id=msg->header.frame_id;
  pt_cloud.header.stamp=msg->header.stamp;
  for(unsigned int i=0;i<msg->joints.size();i++)
    {
      openni_tracker_msgs::jointData new_joint = msg->joints[i];
      Point p;
      p.x=new_joint.pose.position.x;
      p.y=new_joint.pose.position.y;
      p.z=new_joint.pose.position.z;
      pt_cloud.push_back(p);
    }
  pt_cloud.header.frame_id=msg->header.frame_id;
  return pt_cloud;
}
pcl::PointCloud<Point> ConvertPointCloud(pcl::PointCloud<Point> cloud_in, Eigen::Vector3f offset, Eigen::Quaternionf rot)
{
  pcl::PointCloud<Point> cloud_out;
  pcl::transformPointCloud(cloud_in,cloud_out,offset,rot);
  cloud_out.header=cloud_in.header;
  cloud_out.header.frame_id=fixed_frame_id;
  return cloud_out;
}
void PublishPointCloud(pcl::PointCloud<Point> cloud_in,int kinectID)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_in,cloud_msg);
  //ROS_INFO("kinectID: %d",kinectID);
  if(kinectID==0)
    {
      //ROS_INFO("Published");
      skeleton_cloud_pub_0.publish(cloud_msg);
    }
  else if(kinectID==1)
    {
      //ROS_INFO("Published!");
      skeleton_cloud_pub_1.publish(cloud_msg);
    }
}
void msgCallback(const openni_tracker_msgs::skeletonData::ConstPtr &msg)
{
  // Retrieve the data from the two nodes and either save or fusion them.
  ros::Time t=msg->header.stamp;
  ROS_INFO("time=%8f, kinectID: %d",t.toSec(),msg->kinectID);

  pcl::PointCloud<Point> pt_cloud=ExtractPointCloud(msg);
  tf::StampedTransform transf;

  try{
    tf_listener.waitForTransform(msg->header.frame_id, fixed_frame_id,msg->header.stamp, ros::Duration(2.0));
    tf_listener.lookupTransform(msg->header.frame_id, fixed_frame_id,msg->header.stamp, transf);
  }
  catch(tf::TransformException ex)
    {
      ROS_ERROR("SkeletonFusion TF error:%s", ex.what());
      return;
    }    
  tf::Vector3 v3 = transf.getOrigin();
  tf::Quaternion quat = transf.getRotation();
  Eigen::Quaternionf rot(quat.w(), quat.x(), quat.y(), quat.z());
  Eigen::Vector3f offset(v3.x(), v3.y(), v3.z());
 
  pcl::PointCloud<Point> converted_pt_cloud;
  converted_pt_cloud=ConvertPointCloud(pt_cloud,offset,rot);
  PublishPointCloud(converted_pt_cloud,msg->kinectID);

  /*
  for(int k=0;k<=24;k++)// Looping through all the joints
    {
      openni_tracker_msgs::jointData new_joint = msg->joints[k];
      if(new_joint.confidence==0.0)
	{
	  std::cout<<"Zero Conf"<<std::endl;
	  continue;
	}
      person.joints[k].pose.position.x=(person.joints[k].pose.position.x*person.joints[k].confidence+new_joint.pose.position.x*new_joint.confidence)/(new_joint.confidence+person.joints[k].confidence);
      person.joints[k].pose.position.y=(person.joints[k].pose.position.y*person.joints[k].confidence+new_joint.pose.position.y*new_joint.confidence)/(new_joint.confidence+person.joints[k].confidence);
      person.joints[k].pose.position.z=(person.joints[k].pose.position.z*person.joints[k].confidence+new_joint.pose.position.z*new_joint.confidence)/(new_joint.confidence+person.joints[k].confidence);
      // Set new confidence & frame_ID <-----------------------
    }
  
  publishGlobalSkeleton();
  //publishTF();*/
}


void init_variables()
{
  //person.joints=vector<float> values(24);
  openni_tracker_msgs::jointData nj;
  nj.pose.position.x=0.0;
  nj.pose.position.y=0.0;
  nj.pose.position.z=0.0;
  nj.confidence=0.00001;
 for(int k=0;k<=24;k++)// Looping through all the joints
    { 
      nj.jointID=k;
      person.joints.push_back(nj);      
    }
}

};;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "skeleton_fusion");
  skeleton_fusion sk;
  ROS_INFO("Skeleton fusion initialized");
  ros::spin();
}

