#ifndef shape_features_h_DEFINED
#define shape_features_h_DEFINED

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <cpl_visual_features/features/shape_context.h>
#include <geometry_msgs/PointStamped.h>
#include <vector>

class ShapeLocation
{
 public:
  ShapeLocation(geometry_msgs::Point boundary_loc, cpl_visual_features::ShapeDescriptor descriptor) :
      boundary_loc_(boundary_loc), descriptor_(descriptor)
  {
  }
  geometry_msgs::Point boundary_loc_;
  cpl_visual_features::ShapeDescriptor descriptor_;
 private:
  ShapeLocation()
  {
  }
};

typedef std::vector<ShapeLocation> ShapeLocations;

cv::Mat getObjectFootprint(cv::Mat obj_mask, pcl16::PointCloud<pcl16::PointXYZ>& cloud);

ShapeLocations extractFootprintShapeFeature(cv::Mat obj_mask, pcl16::PointCloud<pcl16::PointXYZ>& cloud,
                                            geometry_msgs::Point centroid);
#endif // shape_features_h_DEFINED
