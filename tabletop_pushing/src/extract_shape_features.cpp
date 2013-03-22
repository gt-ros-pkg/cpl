#include <sstream>
#include <iostream>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <ros/ros.h>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <pcl16/io/pcd_io.h>
#include <pcl16_ros/transforms.h>
#include <pcl16/ros/conversions.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <cpl_visual_features/helpers.h>

using namespace cpl_visual_features;
using namespace tabletop_pushing;

typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;

ShapeLocations start_loc_history_;
double start_loc_arc_length_percent_;
int start_loc_push_sample_count_;

// cpl_visual_features::ShapeDescriptor getShapeDescriptor(ProtoObject& cur_obj, pcl16::PointXYZ& start_point)
// {
//   // Get hull_cloud from object_cloud
//   XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj);
//   ShapeLocations locs = tabletop_pushing::extractShapeFeaturesFromSamples(hull_cloud, cur_obj,
//                                                                           true);
//   // TODO: Run through locs and find closest one to push_start_loc, return that descriptor
//   int boundary_loc_idx;
//   double min_dist = FLT_MAX;
//   for (unsigned int i = 0; i < locs.size(); ++i)
//   {
//     double dist_i = pcl_segmenter_->sqrDist(start_point, locs[i].boundary_loc_);
//     if (dist_i < min_dist)
//     {
//       min_dist = dist_i;
//       boundary_loc_idx = i;
//     }
//   }
//   return locs[boundary_loc_idx].descriptor_;
// }

// pcl16::PointXYZ getStartPoint(ProtoObject& cur_obj, float push_angle)
// {
//   // TODO: Get start point using push_angle, init_centroid, and obj_cloud
// }

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

pcl16::PointXYZ objectPointInWorldFrame(pcl16::PointXYZ obj_pt, PushTrackerState& cur_state)
{
  // Rotate out of object frame
  pcl16::PointXYZ rotated_pt;
  double ct = cos(cur_state.x.theta);
  double st = sin(cur_state.x.theta);
  rotated_pt.x = ct*obj_pt.x - st*obj_pt.y;
  rotated_pt.y = st*obj_pt.x + ct*obj_pt.y;
  rotated_pt.z = obj_pt.z;  // NOTE: Currently assume 2D motion
  // Shift to world frame
  pcl16::PointXYZ world_pt;
  world_pt.x = rotated_pt.x + cur_state.x.x;
  world_pt.y = rotated_pt.y + cur_state.x.y;
  world_pt.z = rotated_pt.z + cur_state.z;
  return world_pt;
}

static inline double dist(pcl16::PointXYZ a, pcl16::PointXYZ b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  const double dz = a.z-b.z;
  return std::sqrt(dx*dx+dy*dy+dz*dz);
}

static inline double sqrDist(pcl16::PointXYZ a, pcl16::PointXYZ b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  const double dz = a.z-b.z;
  return dx*dx+dy*dy+dz*dz;
}


ShapeLocation chooseFixedGoalPushStartLoc(ProtoObject& cur_obj, PushTrackerState& cur_state, bool new_object,
                                          int num_start_loc_pushes_per_sample, int num_start_loc_sample_locs)
{
  XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj);

  int rot_idx = -1;
  if (new_object)
  {
    // Reset boundary traversal data
    start_loc_arc_length_percent_ = 0.0;
    start_loc_push_sample_count_ = 0;
    start_loc_history_.clear();

    // NOTE: Initial start location is the dominant orientation
    ROS_INFO_STREAM("Current state theta is: " << cur_state.x.theta);
    double min_angle_dist = FLT_MAX;
    for (int i = 0; i < hull_cloud.size(); ++i)
    {
      double theta_i = atan2(hull_cloud.at(i).y - cur_state.x.y, hull_cloud.at(i).x - cur_state.x.x);
      double angle_dist_i = fabs(subPIAngle(theta_i - cur_state.x.theta));
      if (angle_dist_i < min_angle_dist)
      {
        min_angle_dist = angle_dist_i;
        rot_idx = i;
      }
    }
  }
  else
  {
    // Increment boundary location if necessary
    if (start_loc_history_.size() % num_start_loc_pushes_per_sample == 0)
    {
      start_loc_arc_length_percent_ += 1.0/num_start_loc_sample_locs;
      ROS_INFO_STREAM("Incrementing arc length percent based on: " << num_start_loc_pushes_per_sample);
    }

    // Get initial object boundary location in the current world frame
    ROS_INFO_STREAM("init_obj_point: " << start_loc_history_[0].boundary_loc_);
    pcl16::PointXYZ init_loc_point = objectPointInWorldFrame(start_loc_history_[0].boundary_loc_, cur_state);
    ROS_INFO_STREAM("init_loc_point: " << init_loc_point);

    // Find index of closest point on current boundary to the initial pushing location
    double min_dist = FLT_MAX;
    for (int i = 0; i < hull_cloud.size(); ++i)
    {
      double dist_i = sqrDist(init_loc_point, hull_cloud.at(i));
      if (dist_i < min_dist)
      {
        min_dist = dist_i;
        rot_idx = i;
      }
    }
  }
  // Test hull_cloud orientation, reverse iteration if it is negative
  double pt0_theta = atan2(hull_cloud[rot_idx].y - cur_state.x.y, hull_cloud[rot_idx].x - cur_state.x.x);
  int pt1_idx = (rot_idx+1) % hull_cloud.size();
  double pt1_theta = atan2(hull_cloud[pt1_idx].y - cur_state.x.y, hull_cloud[pt1_idx].x - cur_state.x.x);
  bool reverse_data = false;
  if (subPIAngle(pt1_theta - pt0_theta) < 0)
  {
    reverse_data = true;
    ROS_INFO_STREAM("Reversing data for boundaries");
  }

  // Compute cumulative distance around the boundary at each point
  std::vector<double> boundary_dists(hull_cloud.size(), 0.0);
  double boundary_length = 0.0;
  ROS_INFO_STREAM("rot_idx is " << rot_idx);
  for (int i = 1; i <= hull_cloud.size(); ++i)
  {
    int idx0 = (rot_idx+i-1) % hull_cloud.size();
    int idx1 = (rot_idx+i) % hull_cloud.size();
    if (reverse_data)
    {
      idx0 = (hull_cloud.size()+rot_idx-i+1) % hull_cloud.size();
      idx1 = (hull_cloud.size()+rot_idx-i) % hull_cloud.size();
    }
    // NOTE: This makes boundary_dists[rot_idx] = 0.0, and we have no location at 100% the boundary_length
    boundary_dists[idx0] = boundary_length;
    double loc_dist = dist(hull_cloud[idx0], hull_cloud[idx1]);
    boundary_length += loc_dist;
  }

  // Find location at start_loc_arc_length_percent_ around the boundary
  double desired_boundary_dist = start_loc_arc_length_percent_*boundary_length;
  ROS_INFO_STREAM("Finding location at dist " << desired_boundary_dist << " ~= " << start_loc_arc_length_percent_*100 <<
                  "\% of " << boundary_length);
  int boundary_loc_idx;
  double min_boundary_dist_diff = FLT_MAX;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    double boundary_dist_diff_i = fabs(desired_boundary_dist - boundary_dists[i]);
    if (boundary_dist_diff_i < min_boundary_dist_diff)
    {
      min_boundary_dist_diff = boundary_dist_diff_i;
      boundary_loc_idx = i;
    }
  }

  // Get descriptor at the chosen location
  ShapeLocations locs = tabletop_pushing::extractShapeFeaturesFromSamples(hull_cloud, cur_obj,
                                                                          true);
  // Add into pushing history in object frame
  ShapeLocation s(worldPointInObjectFrame(locs[boundary_loc_idx].boundary_loc_, cur_state),
                  locs[boundary_loc_idx].descriptor_);
  start_loc_history_.push_back(s);

  return locs[boundary_loc_idx];
}

ShapeDescriptor getTrialDescriptor(std::string cloud_path, pcl16::PointXYZ init_loc, float init_theta, bool new_object)
{
  int num_start_loc_pushes_per_sample = 3;
  int num_start_loc_sample_locs = 16;

  // pcl16::PointCloud<pcl16::PointXYZ>::Ptr cloud(new pcl16::PointCloud<pcl16::PointXYZ>);
  ProtoObject cur_obj;
  //.cloud = ; // TODO: COPY from read in one?
  PushTrackerState cur_state;
  cur_state.x.x = init_loc.x;
  cur_state.x.y = init_loc.y;
  cur_state.x.theta = init_theta;
  cur_state.z = init_loc.z;
  cur_obj.centroid[0] = cur_state.x.x;
  cur_obj.centroid[1] = cur_state.x.y;
  cur_obj.centroid[2] = cur_state.z;
  if (pcl16::io::loadPCDFile<pcl16::PointXYZ> (cloud_path, cur_obj.cloud) == -1) //* load the file
  {
    ROS_ERROR_STREAM("Couldn't read file " << cloud_path);
  }
  ShapeLocation sl = chooseFixedGoalPushStartLoc(cur_obj, cur_state, new_object, num_start_loc_pushes_per_sample,
                                                 num_start_loc_sample_locs);
  return sl.descriptor_;
}

class TrialStuff
{
 public:
  TrialStuff(float init_x_, float init_y_, float init_z_, float init_theta_, std::string trial_id_, bool new_object_) :
      init_loc(init_x_, init_y_, init_z_), init_theta(init_theta_), trial_id(trial_id_), new_object(new_object_)
  {
  }
  pcl16::PointXYZ init_loc;
  float init_theta;
  std::string trial_id;
  bool new_object;
};

std::vector<TrialStuff> getTrialsFromFile(std::string aff_file_name)
{
  const std::string trial_header_start = "# object_id/trial_id";
  std::vector<TrialStuff> trials;
  std::ifstream trials_in(aff_file_name.c_str());

  bool next_line_trial = false;
  bool trial_is_start = true;
  int line_count = 0;
  int object_comment = 0;
  bool new_object = true;
  while(trials_in.good())
  {
    char c_line[1024];
    trials_in.getline(c_line, 1024);
    line_count++;
    if (next_line_trial)
    {
      // ROS_INFO_STREAM("Parsing trial_line: ");
      next_line_trial = false;
      // TODO: Parse this line!
      std::stringstream trial_line;
      trial_line << c_line;
      char trial_id_c[256];
      trial_line.getline(trial_id_c, 256, ' ');
      std::stringstream trial_id;
      trial_id << trial_id_c;
      // ROS_INFO_STREAM("Read trial_id: " << trial_id.str());
      float x, y, z, theta;
      trial_line >> x >> y >> z >> theta;
      // ROS_INFO_STREAM("Init pose (" << x << ", " << y << ", " << z << ", " << theta << ")");
      TrialStuff trial(x, y, z, theta, trial_id.str(), new_object);
      trials.push_back(trial);

      new_object = false;
    }
    if (c_line[0] == '#')
    {
      if (c_line[2] == 'o')
      {
        if (trial_is_start)
        {
          next_line_trial = true;
          object_comment++;
        }
        // Switch state
        trial_is_start = !trial_is_start;
      }
      else if (c_line[1] == 'B')
      {
        trial_is_start = true;
        trials.pop_back();
      }
    }
  }
  trials_in.close();
  ROS_INFO_STREAM("Read in: " << line_count << " lines");
  ROS_INFO_STREAM("Read in: " << object_comment << " trial headers");
  ROS_INFO_STREAM("Read in: " << trials.size() << " trials");
  return trials;
}

void writeNewFile(std::string out_file_name, std::vector<TrialStuff> trials, ShapeDescriptors descriptors)
{
  std::ofstream out_file(out_file_name.c_str());
  for (unsigned int i = 0; i < descriptors.size(); ++i)
  {
    for (unsigned int j = 0; j < descriptors[i].size(); ++j)
    {
      out_file << descriptors[i][j] << " ";
    }
    out_file << "\n";
  }
  out_file.close();
}

int main(int argc, char** argv)
{
  // TODO: Get the aff_file and the directory as input
  std::string aff_file_path(argv[1]);
  std::string data_directory_path(argv[2]);
  std::string out_file_name(argv[3]);
  std::vector<TrialStuff> trials = getTrialsFromFile(aff_file_path);

  ShapeDescriptors descriptors;
  // TODO: Read in data file, parse for the below info, then write out new one
  for (unsigned int i = 0; i < trials.size(); ++i)
  {
    std::string trial_id = trials[i].trial_id;
    pcl16::PointXYZ init_loc = trials[i].init_loc;
    float init_theta = trials[i].init_theta;
    bool new_object = trials[i].new_object;
    ROS_INFO_STREAM("trial_id: " << trial_id);
    ROS_INFO_STREAM("init_theta: " << init_theta);
    ROS_INFO_STREAM("init_loc: " << init_loc);
    ROS_INFO_STREAM("new object: " << new_object);
    std::stringstream cloud_path;
    cloud_path << data_directory_path << trial_id << "_obj_cloud.pcd";
    ShapeDescriptor sd = getTrialDescriptor(cloud_path.str(), init_loc, init_theta, new_object);
    std::stringstream descriptor;
    descriptor << "[";
    for (unsigned int i = 0; i < sd.size(); ++i)
    {
      descriptor << " " << sd[i];
    }
    descriptor << "]";
    ROS_INFO_STREAM("Sd is: " << descriptor.str() << "\n");
    descriptors.push_back(sd);
  }
  std::stringstream out_file;
  out_file << data_directory_path << out_file_name;
  writeNewFile(out_file.str(), trials, descriptors);
  return 0;
}
