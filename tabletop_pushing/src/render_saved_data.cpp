// STL
#include <sstream>
#include <iostream>
#include <fstream>
// BOOST
#include <boost/tokenizer.hpp>
// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Pose2D.h>

// PCL
#include <pcl16/io/io.h>
#include <pcl16/io/pcd_io.h>
#include <pcl16/common/centroid.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Ours
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <tabletop_pushing/io_utils.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>
#include <tabletop_pushing/shape_features.h>

using geometry_msgs::Pose2D;
using geometry_msgs::Pose;
using geometry_msgs::Twist;
using namespace tabletop_pushing;

typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;
//
// Simple structs
//
class ControlTimeStep
{
 public:
  ControlTimeStep()
  {
  }
  Pose2D x;
  Pose2D x_dot;
  Pose2D x_desired;
  Twist u;
  double theta0;
  float t;
  Pose ee;
  int seq;
};

typedef std::vector<ControlTimeStep> ControlTrajectory;

class PushTrial
{
 public:
  PushTrial(std::string trial_id_,
            double init_x_, double init_y_, double init_z_, double init_theta_,
            double final_x_, double final_y_, double final_z_, double final_theta_,
            double goal_x, double goal_y, double goal_theta,
            double push_x_, double push_y_, double push_z_,
            std::string primitive_, std::string controller_, std::string proxy_,
            std::string which_arm_, double push_time_, std::string precon,
            double score_) :
      trial_id(trial_id_), init_loc(init_x_, init_y_, init_z_), init_theta(init_theta_),
      final_loc(final_x_, final_y_, final_z_), final_theta(final_theta_),
      start_pt(push_x_, push_y_, push_z_),
      primitive(primitive_), controller(controller_),
      proxy(proxy_), which_arm(which_arm_), push_time(push_time_),
      precondition_method(precon), score(score_)
  {
    goal_pose.x = goal_x;
    goal_pose.y = goal_y;
    goal_pose.theta = goal_theta;
  }

  PushTrial(double init_x_, double init_y_, double init_z_, double init_theta_, std::string trial_id_,
            double push_x_, double push_y_, double push_z_) :
      trial_id(trial_id_), init_loc(init_x_, init_y_, init_z_), init_theta(init_theta_),
      start_pt(push_x_, push_y_, push_z_)
  {
  }
  // Members
  std::string trial_id;
  pcl16::PointXYZ init_loc;
  double init_theta;
  pcl16::PointXYZ final_loc;
  double final_theta;
  Pose2D goal_pose;
  pcl16::PointXYZ start_pt;
  std::string primitive;
  std::string controller;
  std::string proxy;
  std::string which_arm;
  double push_time;
  std::string precondition_method;
  double score;
  ControlTrajectory obj_trajectory;
};

//
// I/O Functions
//
ControlTimeStep parseControlLine(std::stringstream& control_line)
{
  ControlTimeStep cts;
  control_line >> cts.x.x >> cts.x.y >> cts.x.theta;
  control_line >> cts.x_dot.x >> cts.x_dot.y >> cts.x_dot.theta;
  control_line >> cts.x_desired.x >> cts.x_desired.y >> cts.x_desired.theta;
  control_line >> cts.theta0;
  control_line >> cts.u.linear.x >> cts.u.linear.y >> cts.u.linear.z;
  control_line >> cts.u.angular.x >> cts.u.angular.y >> cts.u.angular.z;
  control_line >> cts.t;
  control_line >> cts.ee.position.x >> cts.ee.position.y >> cts.ee.position.z;
  control_line >> cts.ee.orientation.x >> cts.ee.orientation.y >> cts.ee.orientation.z >> cts.ee.orientation.w;
  control_line >> cts.seq;
  return cts;
}

PushTrial parseTrialLine(std::string trial_line_str)
{
  boost::char_separator<char> sep(" ");
  boost::tokenizer<boost::char_separator<char> > tokens(trial_line_str, sep);
  boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();
  std::string trial_id = *it++; // object_id
  // Init loc
  double init_x = boost::lexical_cast<double>(*it++);
  double init_y = boost::lexical_cast<double>(*it++);
  double init_z = boost::lexical_cast<double>(*it++);
  double init_theta = boost::lexical_cast<double>(*it++);
  // Final loc
  double final_x = boost::lexical_cast<double>(*it++);
  double final_y = boost::lexical_cast<double>(*it++);
  double final_z = boost::lexical_cast<double>(*it++);
  double final_theta = boost::lexical_cast<double>(*it++);
  // Goal pose
  double goal_x = boost::lexical_cast<double>(*it++);
  double goal_y = boost::lexical_cast<double>(*it++);
  double goal_theta = boost::lexical_cast<double>(*it++);
  // Push start point
  double push_start_x = boost::lexical_cast<double>(*it++);
  double push_start_y = boost::lexical_cast<double>(*it++);
  double push_start_z = boost::lexical_cast<double>(*it++);
  std::string primitive = *it++; // Primitive
  std::string controller = *it++; // Controller
  std::string proxy = *it++; // Proxy
  std::string which_arm = *it++; // which_arm
  // Push time
  double push_time = boost::lexical_cast<double>(*it++);
  // Precondition
  std::string precon = *it++;
  // score
  double score = boost::lexical_cast<double>(*it++);
  PushTrial trial(trial_id, init_x, init_y, init_z, init_theta,
                  final_x, final_y, final_z, final_theta,
                  goal_x, goal_y, goal_theta,
                  push_start_x, push_start_y, push_start_z,
                  primitive, controller, proxy, which_arm, push_time, precon, score);
  return trial;
}


PushTrial parseTrialLineOld(std::string trial_line_str)
{
  std::stringstream trial_line;
  trial_line << trial_line_str;
  char trial_id_c[4096];
  trial_line.getline(trial_id_c, 4096, ' ');
  std::stringstream trial_id;
  trial_id << trial_id_c;
  double init_x, init_y, init_z, init_theta;
  trial_line >> init_x >> init_y >> init_z >> init_theta;
  double final_x, final_y, final_z, final_theta;
  trial_line >> final_x >> final_y >> final_z >> final_theta;
  double goal_x, goal_y, goal_theta;
  trial_line >> goal_x >> goal_y >> goal_theta;
  double push_start_x, push_start_y, push_start_z, push_start_theta;
  trial_line >> push_start_x >> push_start_y >> push_start_z;
  // TODO: Parse behavior_primitive, controller, proxy, which_arm, push_time,
  // precondition_method, score, shape_descriptors
  PushTrial trial(init_x, init_y, init_z, init_theta, trial_id.str(),
                  push_start_x, push_start_y, push_start_z);
  return trial;
}

std::vector<PushTrial> getTrialsFromFile(std::string aff_file_name)
{
  std::vector<PushTrial> trials;
  std::ifstream trials_in(aff_file_name.c_str());

  bool next_line_trial = false;
  bool trial_is_start = true;
  int line_count = 0;
  int object_comment = 0;
  int trial_starts = 0;
  int bad_stops = 0;
  int good_stops = 0;
  int control_headers = 0;
  bool check_for_control_line = false;
  while(trials_in.good())
  {
    char c_line[4096];
    trials_in.getline(c_line, 4096);
    line_count++;
    if (next_line_trial)
    {
      // ROS_INFO_STREAM("Parsing trial_line: ");
      next_line_trial = false;
      check_for_control_line = false;

      // Parse this line!
      std::stringstream trial_line;
      trial_line << c_line;
      PushTrial trial = parseTrialLine(trial_line.str());
      PushTrial old_trial = parseTrialLineOld(trial_line.str());
      trials.push_back(trial);
    }
    if (c_line[0] == '#')
    {
      if (c_line[2] == 'o')
      {
        check_for_control_line = false;
        object_comment++;
        if (trial_is_start)
        {
          next_line_trial = true;
          trial_starts += 1;
          // ROS_INFO_STREAM("Read in start line");
        }
        else
        {
          // ROS_INFO_STREAM("Read in end line");
          good_stops++;
        }
        // Switch state
        trial_is_start = !trial_is_start;
      }
      else if (c_line[2] == 'x')
      {
        control_headers += 1;
        check_for_control_line = true;
      }
      else if (c_line[1] == 'B')
      {
        // ROS_WARN_STREAM("Read in bad line" << c_line);
        trial_is_start = true;
        trials.pop_back();
        bad_stops += 1;
      }
    }
    else if (check_for_control_line)
    {
      // Parse control line
      std::stringstream control_line;
      control_line << c_line;
      ControlTimeStep cts = parseControlLine(control_line);
      trials.back().obj_trajectory.push_back(cts);
    }
  }
  trials_in.close();

  ROS_INFO_STREAM("Read in: " << line_count << " lines");
  ROS_INFO_STREAM("Read in: " << control_headers << " control headers");
  ROS_INFO_STREAM("Read in: " << object_comment << " trial headers");
  ROS_INFO_STREAM("Classified: " << trial_starts << " as starts");
  ROS_INFO_STREAM("Classified: " << bad_stops << " as bad");
  ROS_INFO_STREAM("Classified: " << good_stops << " as good");
  ROS_INFO_STREAM("Read in: " << trials.size() << " trials");
  return trials;
}

cv::Point projectPointIntoImage(pcl16::PointXYZ pt_in, tf::Transform t, sensor_msgs::CameraInfo cam_info,
                                int num_downsamples=1)
{
  tf::Vector3 pt_in_tf(pt_in.x, pt_in.y, pt_in.z);
  tf::Vector3 cam_pt = t(pt_in_tf);
  cv::Point img_loc;
  img_loc.x = static_cast<int>((cam_info.K[0]*cam_pt.getX() +
                                cam_info.K[2]*cam_pt.getZ()) /
                               cam_pt.getZ());
  img_loc.y = static_cast<int>((cam_info.K[4]*cam_pt.getY() +
                                cam_info.K[5]*cam_pt.getZ()) /
                               cam_pt.getZ());

  for (int i = 0; i < num_downsamples; ++i)
  {
    img_loc.x /= 2;
    img_loc.y /= 2;
  }
  return img_loc;
}


cv::Mat trackerDisplay(cv::Mat& in_frame, PushTrackerState& state, ProtoObject& obj,
                       tf::Transform workspace_to_camera, sensor_msgs::CameraInfo cam_info)
{
  cv::Mat centroid_frame;
  in_frame.copyTo(centroid_frame);
  pcl16::PointXYZ centroid_point(state.x.x, state.x.y, state.z);
  const cv::Point img_c_idx = projectPointIntoImage(centroid_point, workspace_to_camera, cam_info);
  double theta = state.x.theta;

  const float x_min_rad = (std::cos(theta+0.5*M_PI)*0.05);
  const float y_min_rad = (std::sin(theta+0.5*M_PI)*0.05);
  pcl16::PointXYZ table_min_point(centroid_point.x+x_min_rad, centroid_point.y+y_min_rad,
                                  centroid_point.z);
  const float x_maj_rad = (std::cos(theta)*0.15);
  const float y_maj_rad = (std::sin(theta)*0.15);
  pcl16::PointXYZ table_maj_point(centroid_point.x+x_maj_rad, centroid_point.y+y_maj_rad,
                                  centroid_point.z);
  const cv::Point2f img_min_idx = projectPointIntoImage(table_min_point,
                                                        workspace_to_camera, cam_info);
  const cv::Point2f img_maj_idx = projectPointIntoImage(table_maj_point,
                                                        workspace_to_camera, cam_info);
  cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,0),3);
  cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,0,0),3);
  cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,255),1);
  cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,255,0),1);
  cv::Size img_size;
  img_size.width = std::sqrt(std::pow(img_maj_idx.x-img_c_idx.x,2) +
                             std::pow(img_maj_idx.y-img_c_idx.y,2))*2.0;
  img_size.height = std::sqrt(std::pow(img_min_idx.x-img_c_idx.x,2) +
                              std::pow(img_min_idx.y-img_c_idx.y,2))*2.0;
  return centroid_frame;
}

/**
 * Read in the data, render the images and save to disk
 */
int main(int argc, char** argv)
{
  double boundary_hull_alpha = 0.01;
  std::string aff_file_path(argv[1]);
  std::string data_directory_path(argv[2]);
  std::string out_file_path(argv[3]);

  // Read in aff file grouping each trajectory and number of elements
  std::vector<PushTrial> trials = getTrialsFromFile(aff_file_path);
  for (int i = 0; i < trials.size(); ++i)
  {
    std::stringstream cur_transform_name, cur_cam_info_name;
    cur_transform_name << data_directory_path << "workspace_to_cam_" << (i+1) << ".txt";
    cur_cam_info_name << data_directory_path << "cam_info_" << (i+1) << ".txt";
    tf::Transform camera_to_workspace = readTFTransform(cur_transform_name.str());
    // NOTE: Remove inverse, when you get new data
    tf::Transform workspace_to_camera = camera_to_workspace.inverse();
    sensor_msgs::CameraInfo cam_info = readCameraInfo(cur_cam_info_name.str());
    for (int j = 0; j < trials[i].obj_trajectory.size(); ++j)
    {
      ControlTimeStep cts = trials[i].obj_trajectory[j];
      std::stringstream cur_img_name, cur_obj_name;
      cur_img_name << data_directory_path << "feedback_control_input_" << (i+1) << "_" << j << ".png";
      cur_obj_name << data_directory_path << "feedback_control_obj_" << (i+1) << "_" << j << ".pcd";
      cv::Mat cur_base_img = cv::imread(cur_img_name.str());
      XYZPointCloud cur_obj_cloud;
      if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(cur_obj_name.str(), cur_obj_cloud) == -1) //* load the file
      {
        ROS_ERROR_STREAM("Couldn't read file " << cur_obj_name.str());
      }

      // Create proto object
      ProtoObject cur_obj;
      cur_obj.cloud = cur_obj_cloud;
      pcl16::compute3DCentroid(cur_obj_cloud, cur_obj.centroid);

      // Get current tracker state
      PushTrackerState cur_state;
      cur_state.header.frame_id = "/torso_lift_link"; // TODO: get this correctly from disk
      cur_state.header.stamp = ros::Time(cts.t);
      cur_state.header.seq = cts.seq;
      cur_state.x = cts.x;
      cur_state.x_dot = cts.x_dot;
      cur_state.z = trials[i].init_loc.z; // TODO: get this correctly from disk
      cur_state.init_x.x = trials[i].init_loc.x;
      cur_state.init_x.y = trials[i].init_loc.y;
      cur_state.init_x.theta = cts.theta0;
      // TODO: Need these state components still
      cur_state.no_detection = false;
      // cur_state.controller_name = "";
      // cur_state.proxy_name = "";
      // cur_state.behavior_primitive = "";

      // Get object hull
      XYZPointCloud hull_cloud = getObjectBoundarySamples(cur_obj, boundary_hull_alpha);
      cv::Mat boundary_img = visualizeObjectBoundarySamples(hull_cloud, cur_state);

      // Project estimated contact point into image
      pcl16::PointXYZ hand_pt;
      hand_pt.x = cts.ee.position.x;
      hand_pt.y = cts.ee.position.y;
      hand_pt.z = cts.ee.position.z;
      // TODO: Add a fixed amount projection forward using the hand pose (or axis)
      double roll, pitch, yaw;
      tf::Quaternion q;
      tf::quaternionMsgToTF(cts.ee.orientation, q);
      tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
      pcl16::PointXYZ forward_pt; // TODO: Is this dependent on behavior primitive used?
      forward_pt.x = cts.ee.position.x + cos(yaw)*0.01;;
      forward_pt.y = cts.ee.position.y + sin(yaw)*0.01;
      forward_pt.z = cts.ee.position.z;
      cv::Mat hull_cloud_viz = visualizeObjectContactLocation(hull_cloud, cur_state, hand_pt, forward_pt);

      // Show object state
      cv::Mat obj_state_img = trackerDisplay(cur_base_img, cur_state, cur_obj, workspace_to_camera, cam_info);
      // TODO: Write desired output to disk
      cv::imshow("Cur image", cur_base_img);
      cv::imshow("Cur state", obj_state_img);
      // cv::imshow("Boundary image", boundary_img);
      cv::imshow("Contact pt  image", hull_cloud_viz);

      cv::waitKey();
    }
  }
  return 0;
}
