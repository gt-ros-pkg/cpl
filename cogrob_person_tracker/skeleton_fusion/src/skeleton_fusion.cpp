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
  ros::Publisher skeleton_cloud_pub_99;
  ros::Subscriber skeletonData_sub;
  openni_tracker_msgs::skeletonData person;
  tf::TransformListener tf_listener;
  
public:
  ros::NodeHandle nh_;

  skeleton_fusion():
    nh_("~")
  {
    string temp="openni_depth_frame";
    bool useBag;
    nh_.param("useBag", useBag, false);
	if(useBag)
	{
		skeletonData_sub = nh_.subscribe("/skeletonDataBag", 100, &skeleton_fusion::msgCallback,this);
	}else
	{
		skeletonData_sub = nh_.subscribe("/skeleton_data", 100, &skeleton_fusion::msgCallback,this);
	}

    skeletonData_pub=nh_.advertise<openni_tracker_msgs::skeletonData>("/person", 1000);
    skeleton_cloud_pub_0=nh_.advertise<sensor_msgs::PointCloud2>("/skeleton_cloud_0",1);
    skeleton_cloud_pub_1=nh_.advertise<sensor_msgs::PointCloud2>("/skeleton_cloud_1",1);
    skeleton_cloud_pub_99=nh_.advertise<sensor_msgs::PointCloud2>("/skeleton_cloud_99",1);
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
	// This is for rotating the data
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
  std::cout<<"published person!"<<std::endl;
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

pcl::PointCloud<Point> ExtractPointCloud(const openni_tracker_msgs::skeletonData::ConstPtr &msg,  vector<int> &jointID_array,vector<float> &confidences)
{
  pcl::PointCloud<Point> pt_cloud;
  pt_cloud.header.frame_id=msg->header.frame_id;
  pt_cloud.header.stamp=msg->header.stamp;
  for(unsigned int i=0;i<msg->joints.size();i++)
    {
      openni_tracker_msgs::jointData new_joint = msg->joints[i];
      jointID_array.push_back(new_joint.jointID);
      confidences.push_back(new_joint.confidence);
      Point p;
      p.x=new_joint.pose.position.x;
      p.y=new_joint.pose.position.y;
      p.z=new_joint.pose.position.z;
      pt_cloud.push_back(p);
    }
  pt_cloud.header.frame_id=msg->header.frame_id;

  return pt_cloud;
}
pcl::PointCloud<Point> ExtractPointCloudPerson()
{
  pcl::PointCloud<Point> pt_cloud;
  pt_cloud.header.frame_id=person.header.frame_id;
  pt_cloud.header.stamp=person.header.stamp;
  for(unsigned int i=0;i<person.joints.size();i++)
    {
      openni_tracker_msgs::jointData new_joint = person.joints[i];
      Point p;
      p.x=new_joint.pose.position.x;
      p.y=new_joint.pose.position.y;
      p.z=new_joint.pose.position.z;
      pt_cloud.push_back(p);
    }
  return pt_cloud;
}

openni_tracker_msgs::skeletonData ConvertSkeletonData(const openni_tracker_msgs::skeletonData::ConstPtr &msg)
{
  openni_tracker_msgs::skeletonData out_skeleton;
  out_skeleton.header.frame_id=fixed_frame_id;
  out_skeleton.header.stamp=ros::Time::now();
  for(unsigned int i=0;i<msg->joints.size();i++)
    {
      openni_tracker_msgs::jointData new_joint = msg->joints[i];
      geometry_msgs::PointStamped in_pt;
      geometry_msgs::PointStamped out_pt;
      in_pt.header=msg->header;
      in_pt.point.x=new_joint.pose.position.x;
      in_pt.point.y=new_joint.pose.position.y;
      in_pt.point.z=new_joint.pose.position.z;
      std::cout<<"transformPoint"<<std::endl;
      tf_listener.transformPoint(fixed_frame_id, in_pt, out_pt);
      openni_tracker_msgs::jointData new_joint2;
      new_joint2.pose.position.x=out_pt.point.x;
      new_joint2.pose.position.y=out_pt.point.y;
      new_joint2.pose.position.z=out_pt.point.z;      
      out_skeleton.joints.push_back(new_joint2);
    }
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
  else if(kinectID==99)
    {
      //ROS_INFO("Published!");
      skeleton_cloud_pub_99.publish(cloud_msg);
    }
}

int getJointIndex(openni_tracker_msgs::skeletonData sk,int jnt_id)
{
	for(int i=0;i<sk.joints.size();i++)
	{
		if(sk.joints[i].jointID==jnt_id)
		{
			return i;
		}
	}
	return -1;
}
void msgCallback(const openni_tracker_msgs::skeletonData::ConstPtr &msg)
{
  // Retrieve the data from the two nodes and either save or fusion them.
	ros::Time t=msg->header.stamp;
	//ROS_INFO("time=%8f, kinectID: %d",t.toSec(),msg->kinectID);

	vector<int> jointID_array;
	vector<float> confidences;
	pcl::PointCloud<Point> pt_cloud=ExtractPointCloud(msg,jointID_array,confidences);
	tf::StampedTransform transf;

	try{
		tf_listener.waitForTransform(fixed_frame_id, msg->header.frame_id ,ros::Time(0), ros::Duration(2.0));
		tf_listener.lookupTransform(fixed_frame_id, msg->header.frame_id,ros::Time(0), transf);
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


	bool success=false;
	int jnt_id=15;
	for(int k=0;k<jointID_array.size();k++)
    {
		if(jointID_array[k]!=jnt_id) //right hand
		{
		  	//continue;
		}
	
		float conf=confidences[k];
		if(conf<0.9)
		{
		  	continue;
		}
		success=true;
		Point pq=converted_pt_cloud.points[k];
		// This should be getJointIndex(person, k) right ? And then call it in the start of the loop.
		int ind=getJointIndex(person,jnt_id);
		// What's happening here ?
		//ind=0;
		ind = k;
		if(jnt_id==-1) // this joint isn't in the list
		{
			openni_tracker_msgs::jointData new_joint;
			new_joint.pose.position.x=pq.x;
			new_joint.pose.position.y=pq.y;
			new_joint.pose.position.z=pq.z;
			new_joint.jointID=jnt_id; // Should get index from getJointIndex ??
			person.joints.push_back(new_joint);
		}
		else
		{
			openni_tracker_msgs::jointData current_joint = person.joints[ind];
			person.joints[ind].pose.position.x=(current_joint.pose.position.x*current_joint.confidence+pq.x*conf)/(conf+current_joint.confidence);
			person.joints[ind].pose.position.y=(current_joint.pose.position.y*current_joint.confidence+pq.y*conf)/(conf+current_joint.confidence);
			person.joints[ind].pose.position.z=(current_joint.pose.position.z*current_joint.confidence+pq.z*conf)/(conf+current_joint.confidence);	  
		}
    }

  /*
  for(int k=0;k<converted_pt_cloud.points.size();k++)
    {
      Point pq=converted_pt_cloud.points[k];
      int jnt_id=jointID_array[k];
      float conf=confidences[k];
      if(conf<0.01)
	{
	  //std::cout<<"Zero Conf"<<std::endl;
	  continue;
	}
      int ind=getJointIndex(person,jnt_id);

      if(jnt_id==-1) // this joint isn't in the list
	{
	  openni_tracker_msgs::jointData new_joint;
	  new_joint.pose.position.x=pq.x;
	  new_joint.pose.position.y=pq.y;
	  new_joint.pose.position.z=pq.z;
	  person.joints.push_back(new_joint);
	}
      else
	{
	  openni_tracker_msgs::jointData current_joint = person.joints[ind];
	  person.joints[ind].pose.position.x=(current_joint.pose.position.x*current_joint.confidence+pq.x*conf)/(conf+current_joint.confidence);
	  person.joints[ind].pose.position.y=(current_joint.pose.position.y*current_joint.confidence+pq.y*conf)/(conf+current_joint.confidence);
	  person.joints[ind].pose.position.z=(current_joint.pose.position.z*current_joint.confidence+pq.z*conf)/(conf+current_joint.confidence);	  
	}
    }
 
  */

	if(success)
	{
		person.header.stamp=ros::Time::now();
		pcl::PointCloud<Point> pt_cloud2=ExtractPointCloudPerson();
		//pcl::PointCloud<Point> converted_pt_cloud2;
		//converted_pt_cloud2=ConvertPointCloud(pt_cloud2,offset,rot);
		PublishPointCloud(pt_cloud2,99);
		//publishGlobalSkeleton();
	}
}


void init_variables()
{
	//person.joints=vector<float> values(24);
	openni_tracker_msgs::jointData nj;
	nj.pose.position.x=0.0;
	nj.pose.position.y=0.0;
	nj.pose.position.z=0.0;
	nj.confidence=0.0001;
	for(int k=0;k<=24;k++)// Looping through all the joints
	{
	/*
		if(k==15)
		{
			nj.jointID=k;
			person.joints.push_back(nj);
		}
		*/
		
		nj.jointID=k;
		person.joints.push_back(nj);
	}
	person.header.frame_id=fixed_frame_id;
}

};;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "skeleton_fusion");
  skeleton_fusion sk;
  ROS_INFO("Skeleton fusion initialized");
  ros::spin();
}

