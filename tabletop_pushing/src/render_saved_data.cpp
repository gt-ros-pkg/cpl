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
#include <geometry_msgs/PointStamped.h>

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
using geometry_msgs::PointStamped;
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
  double z;
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

  void updateTrialWithFinal(PushTrial& final)
  {
    // Didn't know final pose at start
    final_loc = final.final_loc;
    final_theta = final.final_theta;
    // Didn't know push time at start
    push_time = final.push_time;
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

#define XY_RES 0.00075

inline int worldLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}

inline int objLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}

cv::Point worldPtToImgPt(pcl16::PointXYZ world_pt, double min_x, double max_x,
                         double min_y, double max_y)
{
  cv::Point img_pt(worldLocToIdx(world_pt.x, min_x, max_x),
                   worldLocToIdx(world_pt.y, min_y, max_y));
  return img_pt;
}

pcl16::PointXYZ worldPointInObjectFrame(pcl16::PointXYZ world_pt, PushTrackerState& cur_state)
{
  // Center on object frame
  pcl16::PointXYZ shifted_pt;
  shifted_pt.x = world_pt.x - cur_state.x.x;
  shifted_pt.y = world_pt.y - cur_state.x.y;
  shifted_pt.z = world_pt.z - cur_state.z;
  double ct = cos(cur_state.x.theta);
  double st = sin(cur_state.x.theta);
  // Rotate into correct frame
  pcl16::PointXYZ obj_pt;
  obj_pt.x =  ct*shifted_pt.x + st*shifted_pt.y;
  obj_pt.y = -st*shifted_pt.x + ct*shifted_pt.y;
  obj_pt.z = shifted_pt.z; // NOTE: Currently assume 2D motion
  return obj_pt;
}


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
  control_line >> cts.z;
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

std::vector<PushTrial> getTrialsFromFile(std::string aff_file_name)
{
  std::vector<PushTrial> trials;
  std::ifstream trials_in(aff_file_name.c_str());

  bool next_line_trial = false;
  bool next_trial_is_start = true;
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
      if (next_trial_is_start)
      {
        trials.back().updateTrialWithFinal(trial);
      }
      else
      {
        trials.push_back(trial);
      }
    }
    if (c_line[0] == '#')
    {
      if (c_line[2] == 'o')
      {
        check_for_control_line = false;
        object_comment++;
        if (next_trial_is_start)
        {
          next_line_trial = true;
          trial_starts += 1;
        }
        else
        {
          good_stops++;
          next_line_trial = true;
        }
        // Switch state
        next_trial_is_start = !next_trial_is_start;
      }
      else if (c_line[2] == 'x')
      {
        control_headers += 1;
        check_for_control_line = true;
      }
      else if (c_line[1] == 'B')
      {
        next_trial_is_start = true;
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

  // ROS_INFO_STREAM("Read in: " << line_count << " lines");
  // ROS_INFO_STREAM("Read in: " << control_headers << " control headers");
  // ROS_INFO_STREAM("Read in: " << object_comment << " trial headers");
  // ROS_INFO_STREAM("Classified: " << trial_starts << " as starts");
  // ROS_INFO_STREAM("Classified: " << bad_stops << " as bad");
  // ROS_INFO_STREAM("Classified: " << good_stops << " as good");
  ROS_INFO_STREAM("Read in: " << trials.size() << " trials");
  return trials;
}

cv::Mat visualizeObjectBoundarySamplesGlobal(XYZPointCloud& hull_cloud)
{
  double max_y = 0.475;
  double min_y = -0.475;
  double max_x = 0.75;
  double min_x = 0.45;
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC3, cv::Scalar(255,255,255));

  cv::Scalar kuler_green(51, 178, 0);
  cv::Scalar kuler_red(18, 18, 178);
  cv::Scalar kuler_yellow(25, 252, 255);
  cv::Scalar kuler_blue(204, 133, 20);

  cv::Point prev_img_pt, init_img_pt;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    cv::Point img_pt(cols - objLocToIdx(hull_cloud[i].x, min_x, max_x),
                     objLocToIdx(hull_cloud[i].y, min_y, max_y));
    cv::circle(footprint, img_pt, 1, kuler_blue, 3);
    if (i > 0)
    {
      cv::line(footprint, img_pt, prev_img_pt, kuler_blue, 1);
    }
    else
    {
      init_img_pt.x = img_pt.x;
      init_img_pt.y = img_pt.y;
    }
    prev_img_pt.x = img_pt.x;
    prev_img_pt.y = img_pt.y;
  }
  cv::line(footprint, prev_img_pt, init_img_pt, kuler_blue,1);
  return footprint;
}

void visualizeObjectBoundarySamplesGlobal(XYZPointCloud& hull_cloud, cv::Mat& footprint)
{
  double max_y = 0.475;
  double min_y = -0.475;
  double max_x = 0.75;
  double min_x = 0.45;
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);


  cv::Scalar kuler_green(51, 178, 0);
  cv::Scalar kuler_red(18, 18, 178);
  cv::Scalar kuler_yellow(25, 252, 255);
  cv::Scalar kuler_blue(204, 133, 20);

  cv::Point prev_img_pt, init_img_pt;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    cv::Point img_pt(cols - objLocToIdx(hull_cloud[i].x, min_x, max_x),
                     objLocToIdx(hull_cloud[i].y, min_y, max_y));
    cv::circle(footprint, img_pt, 1, kuler_blue, 3);
    if (i > 0)
    {
      cv::line(footprint, img_pt, prev_img_pt, kuler_blue, 1);
    }
    else
    {
      init_img_pt.x = img_pt.x;
      init_img_pt.y = img_pt.y;
    }
    prev_img_pt.x = img_pt.x;
    prev_img_pt.y = img_pt.y;
  }
  cv::line(footprint, prev_img_pt, init_img_pt, kuler_blue,1);
}


cv::Point projectPointIntoImage(pcl16::PointXYZ pt_in, tf::Transform t, sensor_msgs::CameraInfo cam_info,
                                int num_downsamples=1)
{
  // ROS_INFO_STREAM("Pt in:" << pt_in);
  tf::Vector3 pt_in_tf(pt_in.x, pt_in.y, pt_in.z);
  // ROS_INFO_STREAM("Pt in tf: " << pt_in_tf.getX() << ", " << pt_in_tf.getY() << ", " << pt_in_tf.getZ() << ")");
  tf::Vector3 cam_pt = t(pt_in_tf);
  // ROS_INFO_STREAM("Cam_pt: " << cam_pt.getX() << ", " << cam_pt.getY() << ", " << cam_pt.getZ() << ")");
  cv::Point img_loc;
  img_loc.x = static_cast<int>((cam_info.K[0]*cam_pt.getX() +
                                cam_info.K[2]*cam_pt.getZ()) /
                               cam_pt.getZ());
  img_loc.y = static_cast<int>((cam_info.K[4]*cam_pt.getY() +
                                cam_info.K[5]*cam_pt.getZ()) /
                               cam_pt.getZ());
  // ROS_INFO_STREAM("Img_loc: " << img_loc);
  for (int i = 0; i < num_downsamples; ++i)
  {
    img_loc.x /= 2;
    img_loc.y /= 2;
  }
  // ROS_INFO_STREAM("Img_loc down: " << img_loc);
  return img_loc;
}


cv::Mat displayPushVector(cv::Mat& img, pcl16::PointXYZ& start_point, pcl16::PointXYZ& end_point,
                          tf::Transform workspace_to_camera, sensor_msgs::CameraInfo cam_info)
{
  cv::Mat disp_img;
  img.copyTo(disp_img);
  cv::Scalar kuler_green(51, 178, 0);
  cv::Scalar kuler_red(18, 18, 178);

  cv::Point img_start_point = projectPointIntoImage(start_point, workspace_to_camera, cam_info);
  cv::Point img_end_point = projectPointIntoImage(end_point, workspace_to_camera, cam_info);
  cv::line(disp_img, img_start_point, img_end_point, cv::Scalar(0,0,0),3);
  cv::line(disp_img, img_start_point, img_end_point, kuler_green);
  cv::circle(disp_img, img_end_point, 4, cv::Scalar(0,0,0),3);
  cv::circle(disp_img, img_end_point, 4, kuler_green);
  return disp_img;
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

PushTrackerState generateStateFromData(ControlTimeStep& cts, PushTrial& trial)
{
  PushTrackerState state;
  state.header.frame_id = "/torso_lift_link"; // TODO: get this correctly from disk
  state.header.stamp = ros::Time(cts.t);
  state.header.seq = cts.seq;
  state.x = cts.x;
  state.x_dot = cts.x_dot;
  state.z = cts.z;
  state.init_x.x = trial.init_loc.x;
  state.init_x.y = trial.init_loc.y;
  state.init_x.theta = trial.init_theta;
  state.no_detection = false; // TODO: Get this from disk?
  state.controller_name = trial.controller;
  state.proxy_name = trial.proxy;
  state.behavior_primitive = trial.primitive;
  return state;
}

PushTrackerState generateStartStateFromData(PushTrial& trial)
{
  PushTrackerState state;
  state.header.frame_id = "/torso_lift_link"; // TODO: get this correctly from disk
  // state.header.stamp = ros::Time(trial.t);
  state.header.seq = 0;
  state.x.x = trial.init_loc.x;
  state.x.y = trial.init_loc.y;
  state.x.theta = trial.init_theta;
  state.z = trial.init_loc.z;
  state.x_dot.x = 0;
  state.x_dot.y = 0;
  state.x_dot.theta = 0;
  state.init_x.x = trial.init_loc.x;
  state.init_x.y = trial.init_loc.y;
  state.init_x.theta = trial.init_theta;
  state.no_detection = false; // TODO: Get this from disk?
  state.controller_name = trial.controller;
  state.proxy_name = trial.proxy;
  state.behavior_primitive = trial.primitive;
  return state;
}

ProtoObject generateObjectFromState(XYZPointCloud& obj_cloud)
{
  ProtoObject obj;
  obj.cloud = obj_cloud;
  pcl16::compute3DCentroid(obj_cloud, obj.centroid);
  return obj;
}

cv::Mat projectHandIntoBoundaryImage(ControlTimeStep& cts, PushTrackerState& cur_state,
                                     XYZPointCloud& hull_cloud)
{
  pcl16::PointXYZ hand_pt;
  hand_pt.x = cts.ee.position.x;
  hand_pt.y = cts.ee.position.y;
  hand_pt.z = cts.ee.position.z;
  double roll, pitch, yaw;
  tf::Quaternion q;
  tf::quaternionMsgToTF(cts.ee.orientation, q);
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
  // Add a fixed amount projection forward using the hand pose (or axis)
  // TODO: Get the hand model into here and render the whole fucking thing
  pcl16::PointXYZ forward_pt; // TODO: Is this dependent on behavior primitive used?
  forward_pt.x = cts.ee.position.x + cos(yaw)*0.01;;
  forward_pt.y = cts.ee.position.y + sin(yaw)*0.01;
  forward_pt.z = cts.ee.position.z;
  return visualizeObjectContactLocation(hull_cloud, cur_state, hand_pt, forward_pt);
}


void projectCTSOntoObjectBoundary(cv::Mat& footprint, ControlTimeStep& cts, PushTrial& trial)
{
  PushTrackerState cur_state = generateStateFromData(cts, trial);
  cv::Scalar kuler_green(51, 178, 0);
  cv::Scalar kuler_red(18, 18, 178);
  cv::Scalar kuler_yellow(25, 252, 255);
  cv::Scalar kuler_blue(204, 133, 20);
  cv::Scalar kuler_black(0,0,0);
  double max_y = 0.2;
  double min_y = -0.2;
  double max_x = 0.2;
  double min_x = -0.2;
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);

  pcl16::PointXYZ world_pt;
  world_pt.x = cts.ee.position.x;
  world_pt.y = cts.ee.position.y;
  world_pt.z = cts.ee.position.z;
  pcl16::PointXYZ obj_pt = worldPointInObjectFrame(world_pt, cur_state);
  cv::Point img_pt(cols - objLocToIdx(obj_pt.x, min_x, max_x), objLocToIdx(obj_pt.y, min_y, max_y));

  double roll, pitch, yaw;
  tf::Quaternion q;
  tf::quaternionMsgToTF(cts.ee.orientation, q);
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

  // Add a fixed amount projection forward using the hand pose (or axis)
  pcl16::PointXYZ forward_pt;
  forward_pt.x = cts.ee.position.x + cos(yaw)*0.01;;
  forward_pt.y = cts.ee.position.y + sin(yaw)*0.01;
  forward_pt.z = cts.ee.position.z;
  pcl16::PointXYZ forward_obj_pt = worldPointInObjectFrame(forward_pt, cur_state);
  cv::Point forward_img_pt(cols - objLocToIdx(forward_obj_pt.x, min_x, max_x),
                           objLocToIdx(forward_obj_pt.y, min_y, max_y));

  // Create line pointing in velocity vector direction
  pcl16::PointXYZ vector_world_pt;
  float delta_t = 1.;
  vector_world_pt.x = world_pt.x + cts.u.linear.x*delta_t;
  vector_world_pt.y = world_pt.y + cts.u.linear.y*delta_t;
  vector_world_pt.z = world_pt.z;
  pcl16::PointXYZ vector_obj_pt = worldPointInObjectFrame(vector_world_pt, cur_state);
  cv::Point vector_img_pt(cols - objLocToIdx(vector_obj_pt.x, min_x, max_x),
                          objLocToIdx(vector_obj_pt.y, min_y, max_y));

  // Draw shadows first
  cv::circle(footprint, img_pt, 3, kuler_black, 5);
  cv::line(footprint, img_pt, forward_img_pt, kuler_black, 2);
  cv::line(footprint, img_pt, vector_img_pt, kuler_black, 2);
  // Initial point
  cv::circle(footprint, img_pt, 1, kuler_red, 5);
  // EE Heading vector
  cv::line(footprint, img_pt, forward_img_pt, kuler_red, 1);
  // Velocity vector
  cv::line(footprint, img_pt, vector_img_pt, kuler_green, 1);
}

void projectCTSOntoObjectBoundaryGlobal(cv::Mat& footprint, ControlTimeStep& cts, PushTrial& trial)
{
  // PushTrackerState cur_state = generateStateFromData(cts, trial);
  cv::Scalar kuler_green(51, 178, 0);
  cv::Scalar kuler_red(18, 18, 178);
  cv::Scalar kuler_yellow(25, 252, 255);
  cv::Scalar kuler_blue(204, 133, 20);
  cv::Scalar kuler_black(0,0,0);
  double max_y = 0.475;
  double min_y = -0.475;
  double max_x = 0.75;
  double min_x = 0.45;
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);

  pcl16::PointXYZ world_pt;
  world_pt.x = cts.ee.position.x;
  world_pt.y = cts.ee.position.y;
  world_pt.z = cts.ee.position.z;
  cv::Point img_pt(cols - objLocToIdx(world_pt.x, min_x, max_x), objLocToIdx(world_pt.y, min_y, max_y));

  double roll, pitch, yaw;
  tf::Quaternion q;
  tf::quaternionMsgToTF(cts.ee.orientation, q);
  tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

  // Add a fixed amount projection forward using the hand pose (or axis)
  pcl16::PointXYZ forward_pt;
  forward_pt.x = cts.ee.position.x + cos(yaw)*0.01;;
  forward_pt.y = cts.ee.position.y + sin(yaw)*0.01;
  forward_pt.z = cts.ee.position.z;
  cv::Point forward_img_pt(cols - objLocToIdx(forward_pt.x, min_x, max_x),
                           objLocToIdx(forward_pt.y, min_y, max_y));

  // Create line pointing in velocity vector direction
  pcl16::PointXYZ vector_world_pt;
  float delta_t = 1.;
  vector_world_pt.x = world_pt.x + cts.u.linear.x*delta_t;
  vector_world_pt.y = world_pt.y + cts.u.linear.y*delta_t;
  vector_world_pt.z = world_pt.z;
  cv::Point vector_img_pt(cols - objLocToIdx(vector_world_pt.x, min_x, max_x),
                          objLocToIdx(vector_world_pt.y, min_y, max_y));

  // Draw shadows first
  cv::circle(footprint, img_pt, 3, kuler_black, 5);
  cv::line(footprint, img_pt, forward_img_pt, kuler_black, 2);
  cv::line(footprint, img_pt, vector_img_pt, kuler_black, 2);
  // Initial point
  cv::circle(footprint, img_pt, 1, kuler_red, 5);
  // EE Heading vector
  cv::line(footprint, img_pt, forward_img_pt, kuler_red, 1);
  // Velocity vector
  cv::line(footprint, img_pt, vector_img_pt, kuler_green, 1);
}



int main_visualize_training_samples(int argc, char** argv)
{
  double boundary_hull_alpha = 0.01;
  std::string aff_file_path(argv[1]);
  std::string data_directory_path(argv[2]);
  std::string out_file_path(argv[3]);

  int feedback_idx = 1;
  if (argc > 4)
  {
    feedback_idx = atoi(argv[4]);
  }

  int base_img_idx = 0;
  if (argc > 5)
  {
    base_img_idx = atoi(argv[5]);
  }

  int wait_time = 0;
  // Read in aff_file
  std::vector<PushTrial> trials = getTrialsFromFile(aff_file_path);

  if (trials.size() < 1)
  {
    ROS_ERROR_STREAM("No trial data read");
    return -1;
  }

  // Setup base point cloud image
  // HACK: We can specify this to get a prettier image than 0
  std::stringstream base_obj_name;
  base_obj_name << data_directory_path << "feedback_control_obj_" << feedback_idx+base_img_idx << "_"
                << 0 << ".pcd";
  XYZPointCloud base_obj_cloud;
  if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(base_obj_name.str(), base_obj_cloud) == -1) //* load the file
  {
    ROS_ERROR_STREAM("Couldn't read file " << base_obj_name.str());
  }
  ProtoObject base_obj = generateObjectFromState(base_obj_cloud);
  XYZPointCloud base_hull_cloud = getObjectBoundarySamples(base_obj, boundary_hull_alpha);
  PushTrial base_trial = trials[base_img_idx];
  PushTrackerState base_state = generateStartStateFromData(base_trial);
  cv::Mat sample_pt_cloud_combined_img = visualizeObjectBoundarySamples(base_hull_cloud, base_state);

  // Go through all trials reading data associated with them
  for (unsigned int i = 0; i < trials.size(); ++i, feedback_idx++)
  {
    PushTrial trial = trials[i];

    // Read in workspace transform and camera parameters
    std::stringstream trial_transform_name, trial_cam_info_name;
    trial_transform_name << data_directory_path << "workspace_to_cam_" << feedback_idx << ".txt";
    trial_cam_info_name << data_directory_path << "cam_info_" << feedback_idx << ".txt";
    tf::Transform workspace_to_camera = readTFTransform(trial_transform_name.str());
    sensor_msgs::CameraInfo cam_info = readCameraInfo(trial_cam_info_name.str());

    // Read in image and point cloud for object
    // Get associated image and object point cloud for this trial
    std::stringstream trial_img_name, trial_obj_name;
    trial_img_name << data_directory_path << "feedback_control_input_" << feedback_idx << "_" << 0 << ".png";
    trial_obj_name << data_directory_path << "feedback_control_obj_" << feedback_idx << "_" << 0 << ".pcd";
    cv::Mat trial_base_img = cv::imread(trial_img_name.str());
    XYZPointCloud trial_obj_cloud;
    if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(trial_obj_name.str(), trial_obj_cloud) == -1) //* load the file
    {
      ROS_ERROR_STREAM("Couldn't read file " << trial_obj_name.str());
    }

    // Create objects for use by standard methods
    ProtoObject trial_obj = generateObjectFromState(trial_obj_cloud);
    PushTrackerState trial_state = generateStartStateFromData(trial);
    XYZPointCloud hull_cloud = getObjectBoundarySamples(trial_obj, boundary_hull_alpha);
    cv::Mat sample_pt_cloud_img = visualizeObjectBoundarySamples(hull_cloud, trial_state);
    // Go through each control time step in the current trial
    for (unsigned int j = 0; j < trial.obj_trajectory.size(); ++j)
    {
      ControlTimeStep cts = trial.obj_trajectory[j];
      projectCTSOntoObjectBoundary(sample_pt_cloud_img, cts, trial);
      projectCTSOntoObjectBoundary(sample_pt_cloud_combined_img, cts, base_trial);
    }
    ROS_INFO_STREAM("Showing image for trial: " << i);
    cv::imshow("Sample pt image", sample_pt_cloud_img);
    cv::imshow("Combined Sample pt image", sample_pt_cloud_combined_img);
    std::stringstream trial_cloud_out_name;
    trial_cloud_out_name << out_file_path << "trial_samples_" << i << ".png";
    cv::imwrite(trial_cloud_out_name.str(), sample_pt_cloud_img);
    cv::waitKey(wait_time);
  }
  cv::imshow("Combined Sample pt image", sample_pt_cloud_combined_img);
  std::stringstream combined_cloud_out_name;
  combined_cloud_out_name << out_file_path << "combined_samples.png";
  cv::imwrite(combined_cloud_out_name.str(), sample_pt_cloud_combined_img);
  return 0;
}

int main_visualize_training_samples_global(int argc, char** argv)
{
  double boundary_hull_alpha = 0.01;
  std::string aff_file_path(argv[1]);
  std::string data_directory_path(argv[2]);
  std::string out_file_path(argv[3]);

  int feedback_idx = 1;
  if (argc > 4)
  {
    feedback_idx = atoi(argv[4]);
  }

  int base_img_idx = 0;
  if (argc > 5)
  {
    base_img_idx = atoi(argv[5]);
  }

  int wait_time = 0;
  // Read in aff_file
  std::vector<PushTrial> trials = getTrialsFromFile(aff_file_path);

  if (trials.size() < 1)
  {
    ROS_ERROR_STREAM("No trial data read");
    return -1;
  }

  // Setup base point cloud image
  // HACK: We can specify this to get a prettier image than 0
  std::stringstream base_obj_name;
  base_obj_name << data_directory_path << "feedback_control_obj_" << feedback_idx+base_img_idx << "_"
                << 0 << ".pcd";
  XYZPointCloud base_obj_cloud;
  if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(base_obj_name.str(), base_obj_cloud) == -1) //* load the file
  {
    ROS_ERROR_STREAM("Couldn't read file " << base_obj_name.str());
  }
  ProtoObject base_obj = generateObjectFromState(base_obj_cloud);
  XYZPointCloud base_hull_cloud = getObjectBoundarySamples(base_obj, boundary_hull_alpha);
  PushTrial base_trial = trials[base_img_idx];
  PushTrackerState base_state = generateStartStateFromData(base_trial);
  cv::Mat sample_pt_cloud_combined_img = visualizeObjectBoundarySamplesGlobal(base_hull_cloud);

  // Go through all trials reading data associated with them
  for (unsigned int i = 0; i < trials.size(); ++i, feedback_idx++)
  {
    PushTrial trial = trials[i];

    // Read in workspace transform and camera parameters
    std::stringstream trial_transform_name, trial_cam_info_name;
    trial_transform_name << data_directory_path << "workspace_to_cam_" << feedback_idx << ".txt";
    trial_cam_info_name << data_directory_path << "cam_info_" << feedback_idx << ".txt";
    tf::Transform workspace_to_camera = readTFTransform(trial_transform_name.str());
    sensor_msgs::CameraInfo cam_info = readCameraInfo(trial_cam_info_name.str());

    // Read in image and point cloud for object
    // Get associated image and object point cloud for this trial
    std::stringstream trial_img_name, trial_obj_name;
    trial_img_name << data_directory_path << "feedback_control_input_" << feedback_idx << "_" << 0 << ".png";
    trial_obj_name << data_directory_path << "feedback_control_obj_" << feedback_idx << "_" << 0 << ".pcd";
    cv::Mat trial_base_img = cv::imread(trial_img_name.str());
    XYZPointCloud trial_obj_cloud;
    if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(trial_obj_name.str(), trial_obj_cloud) == -1) //* load the file
    {
      ROS_ERROR_STREAM("Couldn't read file " << trial_obj_name.str());
    }

    // Create objects for use by standard methods
    ProtoObject trial_obj = generateObjectFromState(trial_obj_cloud);
    XYZPointCloud hull_cloud = getObjectBoundarySamples(trial_obj, boundary_hull_alpha);
    cv::Mat sample_pt_cloud_img = visualizeObjectBoundarySamplesGlobal(hull_cloud);
    // Go through each control time step in the current trial
    for (unsigned int j = 0; j < trial.obj_trajectory.size(); ++j)
    {
      std::stringstream step_obj_name;
      step_obj_name << data_directory_path << "feedback_control_obj_" << feedback_idx << "_" << j << ".pcd";
      cv::Mat trial_base_img = cv::imread(trial_img_name.str());
      XYZPointCloud step_obj_cloud;
      if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(step_obj_name.str(), step_obj_cloud) == -1) //* load the file
      {
        ROS_ERROR_STREAM("Couldn't read file " << step_obj_name.str());
      }

      ProtoObject step_obj = generateObjectFromState(step_obj_cloud);
      XYZPointCloud step_cloud = getObjectBoundarySamples(step_obj, boundary_hull_alpha);
      // visualizeObjectBoundarySamplesGlobal(step_cloud, sample_pt_cloud_img);
      // visualizeObjectBoundarySamplesGlobal(step_cloud, sample_pt_cloud_combined_img);
      ControlTimeStep cts = trial.obj_trajectory[j];
      projectCTSOntoObjectBoundaryGlobal(sample_pt_cloud_img, cts, trial);
      projectCTSOntoObjectBoundaryGlobal(sample_pt_cloud_combined_img, cts, base_trial);
    }
    ROS_INFO_STREAM("Showing image for trial: " << i);
    cv::imshow("Sample pt image", sample_pt_cloud_img);
    cv::imshow("Combined Sample pt image", sample_pt_cloud_combined_img);
    std::stringstream trial_cloud_out_name;
    trial_cloud_out_name << out_file_path << "trial_samples_" << i << ".png";
    cv::imwrite(trial_cloud_out_name.str(), sample_pt_cloud_img);
    // cv::waitKey(wait_time);
  }
  cv::imshow("Combined Sample pt image", sample_pt_cloud_combined_img);
  std::stringstream combined_cloud_out_name;
  combined_cloud_out_name << out_file_path << "combined_samples.png";
  cv::imwrite(combined_cloud_out_name.str(), sample_pt_cloud_combined_img);
  return 0;
}

/**
 * Read in the data, render the images and save to disk
 */
int main_render(int argc, char** argv)
{

  if (argc < 4 || argc > 6)
  {
    std::cout << "usage: " << argv[0] << " aff_file_path data_directory_path out_file_path [wait_time] [start_idx]" << std::endl;
    return -1;
  }

  double boundary_hull_alpha = 0.01;
  std::string aff_file_path(argv[1]);
  std::string data_directory_path(argv[2]);
  std::string out_file_path(argv[3]);
  double wait_time = 0;
  if (argc > 4)
  {
    wait_time = atoi(argv[4]);
  }

  int feedback_idx = 1;
  if (argc > 5)
  {
    feedback_idx = atoi(argv[5]);
  }

  // Read in aff file grouping each trajectory and number of elements
  std::vector<PushTrial> trials = getTrialsFromFile(aff_file_path);

  // Go through all trials reading data associated with them
  for (unsigned int i = 0; i < trials.size(); ++i, feedback_idx++)
  {
    // Read in workspace transform and camera parameters
    std::stringstream cur_transform_name, cur_cam_info_name;
    cur_transform_name << data_directory_path << "workspace_to_cam_" << feedback_idx << ".txt";
    cur_cam_info_name << data_directory_path << "cam_info_" << feedback_idx << ".txt";
    tf::Transform workspace_to_camera = readTFTransform(cur_transform_name.str());
    sensor_msgs::CameraInfo cam_info = readCameraInfo(cur_cam_info_name.str());

    PushTrial trial = trials[i];
    // Go through each control time step in the current trial
    for (unsigned int j = 0; j < trial.obj_trajectory.size(); ++j)
    {
      ControlTimeStep cts = trial.obj_trajectory[j];

      // Get associated image and object point cloud for this time step
      std::stringstream cur_img_name, cur_obj_name;
      cur_img_name << data_directory_path << "feedback_control_input_" << feedback_idx << "_" << j << ".png";
      cur_obj_name << data_directory_path << "feedback_control_obj_" << feedback_idx << "_" << j << ".pcd";
      cv::Mat cur_base_img = cv::imread(cur_img_name.str());
      XYZPointCloud cur_obj_cloud;
      if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(cur_obj_name.str(), cur_obj_cloud) == -1) //* load the file
      {
        ROS_ERROR_STREAM("Couldn't read file " << cur_obj_name.str());
      }

      // Create objects for use by standard methods
      ProtoObject cur_obj = generateObjectFromState(cur_obj_cloud);
      PushTrackerState cur_state = generateStateFromData(cts, trial);
      XYZPointCloud hull_cloud = getObjectBoundarySamples(cur_obj, boundary_hull_alpha);

      // NOTE: Add any desired images to render below here

      // Project estimated contact point into image
      cv::Mat hull_cloud_viz = projectHandIntoBoundaryImage(cts, cur_state, hull_cloud);
      // Show object state
      cv::Mat obj_state_img = trackerDisplay(cur_base_img, cur_state, cur_obj, workspace_to_camera, cam_info);
      pcl16::PointXYZ start_point;
      start_point.x = cur_state.x.x;
      start_point.y = cur_state.x.y;
      start_point.z = cur_state.z;
      pcl16::PointXYZ goal_point;
      goal_point.x = trial.goal_pose.x;
      goal_point.y = trial.goal_pose.y;
      goal_point.z = cur_state.z;

      cv::Mat goal_vector_img = displayPushVector(cur_base_img, start_point, goal_point,
                                                  workspace_to_camera, cam_info);

      // Display & write desired output to disk
      // cv::imshow("Cur state", obj_state_img);
      // cv::imshow("Contact pt  image", hull_cloud_viz);
      std::stringstream state_out_name, contact_pt_name, goal_out_name;
      state_out_name << out_file_path << "state_" << feedback_idx << "_" << j << ".png";
      contact_pt_name << out_file_path << "contact_pt_" << feedback_idx << "_" << j << ".png";
      goal_out_name << out_file_path << "goal_vector_"  << feedback_idx << "_" << j << ".png";
      cv::imwrite(state_out_name.str(), obj_state_img);
      cv::imwrite(contact_pt_name.str(), hull_cloud_viz);
      cv::imwrite(goal_out_name.str(), goal_vector_img);
      // cv::waitKey(wait_time);
    }
  }
  return 0;
}

int main(int argc, char** argv)
{
  // return main_visualize_training_samples(argc, argv);
  return main_visualize_training_samples_global(argc, argv);
  // return main_render(argc, argv);
}
