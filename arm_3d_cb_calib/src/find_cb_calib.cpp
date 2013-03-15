#include "arm_3d_cb_calib/calib_3d_cbs.h"

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>

#include "geometry_msgs/PoseStamped.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

using namespace arm_3d_cb_calib;

typedef pcl::PointCloud<pcl::PointXYZ> PCXYZ;
typedef pcl::PointXYZ PXYZ;

void readCBPoseBag(char* filename, vector<PCXYZ::Ptr> &pcs, 
                   vector<gtsam::Pose3> &poses, int num)
{
  printf("Reading bag...");
  rosbag::Bag bag_in(filename, rosbag::bagmode::Read);
  vector<string> topics;
  string pc_topic = "/pc";
  string pose_topic = "/pose";
  topics.push_back(pc_topic);
  topics.push_back(pose_topic);
  rosbag::View view(bag_in, rosbag::TopicQuery(topics));
  num *= 2;
  BOOST_FOREACH(rosbag::MessageInstance const m, view) {
    if(m.getTopic() == pc_topic) {
      PCXYZ::Ptr new_pc(new PCXYZ());
      pcl::fromROSMsg<PXYZ>(*m.instantiate<sensor_msgs::PointCloud2>(),*new_pc);
      pcs.push_back(new_pc);
    }
    if(m.getTopic() == pose_topic) {
      geometry_msgs::PoseStamped::ConstPtr pose = m.instantiate<geometry_msgs::PoseStamped>();
      poses.push_back(geomPoseToGtsamPose3(pose->pose));
    }
    if(num >= 0)
      if(--num == 0)
        break;
  }
  bag_in.close();
  printf("done.\n");
}

int myrandom (int i) { return std::rand()%i;}
int main(int argc, char* argv[])
{
  ros::init(argc, argv, "find_cb_calib");
  int num_kinects = argc-1;

  double dim_cb;
  int num_rows, num_cols;
  ros::param::param<double>("~dim_cb", dim_cb, 0.0);
  ros::param::param<int>("~num_rows", num_rows, 7);
  ros::param::param<int>("~num_cols", num_cols, 6);
  CBCalibProblem prob(num_kinects, dim_cb, num_rows, num_cols);
  cout << "dim_cb " << dim_cb << ", num_rows " << num_rows << ", num_cols " << num_cols << endl;

  for(int k=0;k<num_kinects;k++) {
    vector<PCXYZ::Ptr> pcs;
    readCBPoseBag(argv[k+1], pcs, prob.base_T_ee_poses, -1);

    for(size_t i=0;i<pcs.size();i++) {
      prob.kinect_p_points[k].push_back(vector<Point3>());
      for(size_t j=0;j<pcs[i]->size();j++) {
        PXYZ pt = pcs[i]->at(j);
        prob.kinect_p_points[k][i].push_back(Point3(pt.x,pt.y,pt.z));
      }
    }
  }
  
  CBCalibSolution sol;
  //generateCBCalibProblem(prob, ground_sol, 10);
  solveCBCalibProblem(prob, sol);
  //for(int i=0;i<num_kinects;i++)
  //  (sol.kinect_T_base_poses[i].inverse() * ground_sol.kinect_T_base_poses[i]).print();
  //(sol.cb_T_ee_pose.inverse() * ground_sol.cb_T_ee_pose).print();
  sol.cb_T_ee_pose.inverse().print("Checkerboard offset\n");
  Vector3 p = sol.cb_T_ee_pose.translation().vector();
  Quaternion q = sol.cb_T_ee_pose.rotation().toQuaternion();
  printf("Offset: %4f %4f %4f %4f %4f %4f %4f\n", p[0], p[1], p[2], q.x(), q.y(), q.z(), q.w());
  printf("Kinect poses in base frame as Trans-xyz, Quaternion-xyzw\n");

  for(int k=0;k<num_kinects;k++) {
    p = sol.kinect_T_base_poses[k].inverse().translation().vector();
    q = sol.kinect_T_base_poses[k].inverse().rotation().toQuaternion();
    printf("Kinect %d: %4f %4f %4f %4f %4f %4f %4f\n", k+1, p[0], p[1], p[2], q.x(), q.y(), q.z(), q.w());
  }

  return 0;
}
