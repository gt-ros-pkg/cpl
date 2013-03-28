#ifndef shape_features_h_DEFINED
#define shape_features_h_DEFINED

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <cpl_visual_features/features/shape_context.h>
#include <vector>
#include <tabletop_pushing/point_cloud_segmentation.h>

namespace tabletop_pushing
{

class ShapeLocation
{
 public:
  ShapeLocation(pcl16::PointXYZ boundary_loc, cpl_visual_features::ShapeDescriptor descriptor):
      boundary_loc_(boundary_loc), descriptor_(descriptor)
  {
  }
  pcl16::PointXYZ boundary_loc_;
  cpl_visual_features::ShapeDescriptor descriptor_;

  ShapeLocation() : boundary_loc_(0.0,0.0,0.0), descriptor_()
  {
  }
};

typedef std::vector<ShapeLocation> ShapeLocations;

// int getHistBinIdx(int x_idx, int y_idx, int n_x_bins, int n_y_bins);

cv::Mat getObjectFootprint(cv::Mat obj_mask, pcl16::PointCloud<pcl16::PointXYZ>& cloud);

void getPointRangesXY(XYZPointCloud& samples, cpl_visual_features::ShapeDescriptor& sd);

void getCovarianceXYFromPoints(XYZPointCloud& pts, cpl_visual_features::ShapeDescriptor& sd);

pcl16::PointCloud<pcl16::PointXYZ> getObjectBoundarySamples(ProtoObject& cur_obj, double hull_alpha = 0.01);

ShapeLocations extractObjectShapeContext(ProtoObject& cur_obj, bool use_center = true);

ShapeLocations extractShapeContextFromSamples(pcl16::PointCloud<pcl16::PointXYZ>& samples_pcl,
                                              ProtoObject& cur_obj, bool use_center);

cpl_visual_features::ShapeDescriptor extractPointHistogramXY(pcl16::PointCloud<pcl16::PointXYZ>& samples,
                                                             float x_res, float y_res);

pcl16::PointCloud<pcl16::PointXYZ> getLocalSamples(pcl16::PointCloud<pcl16::PointXYZ>& samples_pcl,
                                                   ProtoObject& cur_obj, pcl16::PointXYZ sample_loc, float s,
                                                   float hull_alpha);

cpl_visual_features::ShapeDescriptor extractLocalShapeFeatures(pcl16::PointCloud<pcl16::PointXYZ>& samples_pcl,
                                                               ProtoObject& cur_obj, pcl16::PointXYZ sample_loc,
                                                               float s, float hull_alpha, float hist_res);

cv::Mat computeShapeFeatureAffinityMatrix(ShapeLocations& locs, bool use_center = false);

double shapeFeatureChiSquareDist(cpl_visual_features::ShapeDescriptor& a, cpl_visual_features::ShapeDescriptor& b);

double shapeFeatureSquaredEuclideanDist(cpl_visual_features::ShapeDescriptor& a, cpl_visual_features::ShapeDescriptor& b);

void clusterShapeFeatures(ShapeLocations& locs, int k, std::vector<int>& cluster_ids,
                          cpl_visual_features::ShapeDescriptors& centers, double min_err_change, int max_iter,
                          int num_retries = 5);

int closestShapeFeatureCluster(cpl_visual_features::ShapeDescriptor& descriptor,
                               cpl_visual_features::ShapeDescriptors& centers, double& min_dist);
};
#endif // shape_features_h_DEFINED
