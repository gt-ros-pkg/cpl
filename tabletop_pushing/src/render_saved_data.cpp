// STL
#include <sstream>
#include <iostream>
#include <fstream>
// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Pose2D.h>
// PCL
#include <pcl16/io/io.h>
#include <pcl16/io/pcd_io.h>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tabletop_pushing/point_cloud_segmentation.h>
#include <tabletop_pushing/io_utils.h>

using geometry_msgs::Pose2D;
using geometry_msgs::Pose;
using geometry_msgs::Twist;
using namespace tabletop_pushing;

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
  PushTrial(double init_x_, double init_y_, double init_z_, double init_theta_, std::string trial_id_,
            bool new_object_, double push_x_, double push_y_, double push_z_) :
      init_loc(init_x_, init_y_, init_z_), init_theta(init_theta_), trial_id(trial_id_), new_object(new_object_),
      start_pt(push_x_, push_y_, push_z_)
  {
  }
  pcl16::PointXYZ init_loc;
  double init_theta;
  std::string trial_id;
  bool new_object;
  pcl16::PointXYZ start_pt;
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
  bool new_object = true;
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
      char trial_id_c[4096];
      trial_line.getline(trial_id_c, 4096, ' ');
      std::stringstream trial_id;
      trial_id << trial_id_c;
      // ROS_INFO_STREAM("Read trial_id: " << trial_id.str());
      double init_x, init_y, init_z, init_theta;
      trial_line >> init_x >> init_y >> init_z >> init_theta;
      double final_x, final_y, final_z, final_theta;
      trial_line >> final_x >> final_y >> final_z >> final_theta;
      double goal_x, goal_y, goal_theta;
      trial_line >> goal_x >> goal_y >> goal_theta;
      double push_start_x, push_start_y, push_start_z, push_start_theta;
      trial_line >> push_start_x >> push_start_y >> push_start_z;
      new_object = !trials.size();
      PushTrial trial(init_x, init_y, init_z, init_theta, trial_id.str(), new_object,
                      push_start_x, push_start_y, push_start_z);
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

cv::Mat renderGoalVector(cv::Mat& color_img, ControlTimeStep& cts)
{
  cv::Mat goal_img;
  color_img.copyTo(goal_img);
  return goal_img;
}

/**
 * Read in the data, render the images and save to disk
 */
int main(int argc, char** argv)
{
  std::string aff_file_path(argv[1]);
  std::string data_directory_path(argv[2]);
  std::string out_file_path(argv[3]);

  // Read in aff file grouping each trajectory and number of elements
  std::vector<PushTrial> trials = getTrialsFromFile(aff_file_path);
  for (int i = 0; i < trials.size(); ++i)
  {
    for (int j = 0; j < trials[i].obj_trajectory.size(); ++j)
    {
      std::stringstream cur_img_name, cur_obj_name, cur_transform_name, cur_cam_info_name;
      cur_img_name << data_directory_path << "feedback_control_input_" << (i+1) << "_" << j << ".png";
      cur_obj_name << data_directory_path << "feedback_control_obj_" << (i+1) << "_" << j << ".pcd";
      cur_transform_name << data_directory_path << "workspace_to_cam_" << (i+1) << "_" << j << ".txt";
      cur_cam_info_name << data_directory_path << "cam_info_" << (i+1) << "_" << j << ".txt";
      cv::Mat cur_base_img = cv::imread(cur_img_name.str());
      XYZPointCloud cur_obj_cloud;
      if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(cur_obj_name.str(), cur_obj_cloud) == -1) //* load the file
      {
        ROS_ERROR_STREAM("Couldn't read file " << cur_obj_name.str());
      }
      tf::Transform workspace_to_camera = readTFTransform(cur_transform_name.str());
      sensor_msgs::CameraInfo cam_info = readCameraInfo(cur_cam_info_name.str());
      cv::imshow("Cur image", cur_base_img);
      cv::waitKey();

      // TODO: Operate on images / clouds and produced desired output
      // TODO: Write output to disk
    }
  }
  return 0;
}