#ifndef io_utils_h_DEFINED
#define io_utils_h_DEFINED 1

#include <string>
#include <fstream>
#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <ros/ros.h>

namespace tabletop_pushing
{
void writeTFTransform(tf::StampedTransform& transform, std::string path)
{
  tf::Vector3 trans = transform.getOrigin();
  tf::Quaternion rot = transform.getRotation();
  std::ofstream transform_file(path.c_str());
  transform_file << rot.getX() << " " << rot.getY() << " " << rot.getZ() << " " << rot.getW() << "\n";
  transform_file << trans.getX() << " " << trans.getY() << " " << trans.getZ() << "\n";
  transform_file.close();
}

tf::Transform readTFTransform(std::string path)
{
  std::ifstream transform_file(path.c_str());
  float q_x, q_y, q_z, q_w;
  transform_file >> q_x >> q_y >> q_z >> q_w;
  float t_x, t_y, t_z;
  transform_file >> t_x >> t_y >> t_z;
  transform_file.close();
  tf::Quaternion rot(q_x, q_y, q_z, q_w);
  tf::Vector3 trans(t_x, t_y, t_z);
  tf::Transform t(rot, trans);
  return t;
}

void writeCameraInfo(sensor_msgs::CameraInfo& info, std::string path)
{
  std::ofstream cam_info_file(path.c_str());
  for (int i = 0; i < 9; ++i)
  {
    cam_info_file << info.K[i] << " ";
  }
  cam_info_file.close();
}

sensor_msgs::CameraInfo readCameraInfo(std::string path)
{
  sensor_msgs::CameraInfo info;
  std::ifstream cam_info_file(path.c_str());
  for (int i = 0; i < 9; ++i)
  {
    double ki;
    cam_info_file >> ki;
    info.K[i] = ki;
  }
  cam_info_file.close();
  return info;
}
};
#endif // io_utils_h_DEFINED
