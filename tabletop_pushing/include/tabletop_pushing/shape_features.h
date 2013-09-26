#ifndef shape_features_h_DEFINED
#define shape_features_h_DEFINED

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <cpl_visual_features/features/shape_context.h>
#include <vector>
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>

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

cv::Mat getObjectFootprint(cv::Mat obj_mask, XYZPointCloud& cloud);

void getPointRangesXY(XYZPointCloud& samples, cpl_visual_features::ShapeDescriptor& sd);

void getCovarianceXYFromPoints(XYZPointCloud& pts, cpl_visual_features::ShapeDescriptor& sd);

void extractPCAFeaturesXY(XYZPointCloud& samples, cpl_visual_features::ShapeDescriptor& sd);

void extractBoundingBoxFeatures(XYZPointCloud& samples, cpl_visual_features::ShapeDescriptor& sd);

XYZPointCloud getObjectBoundarySamples(ProtoObject& cur_obj, double hull_alpha = 0.01);

cv::Mat visualizeObjectBoundarySamples(XYZPointCloud& hull_cloud, tabletop_pushing::VisFeedbackPushTrackingFeedback& cur_state);

ShapeLocations extractObjectShapeContext(ProtoObject& cur_obj, bool use_center = true);

ShapeLocations extractShapeContextFromSamples(XYZPointCloud& samples_pcl,
                                              ProtoObject& cur_obj, bool use_center);

XYZPointCloud transformSamplesIntoSampleLocFrame(XYZPointCloud& samples, ProtoObject& cur_obj,
                                                 pcl16::PointXYZ sample_pt);

cpl_visual_features::ShapeDescriptor extractPointHistogramXY(XYZPointCloud& samples, double x_res, double y_res, double x_range,
                                                             double y_range);


XYZPointCloud getLocalSamples(XYZPointCloud& samples_pcl, ProtoObject& cur_obj, pcl16::PointXYZ sample_loc,
                              double s, double hull_alpha);

cpl_visual_features::ShapeDescriptors extractLocalAndGlobalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj,
                                                                        double sample_spread, double hull_alpha,
                                                                        double hist_res);

cpl_visual_features::ShapeDescriptor extractLocalAndGlobalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj,
                                                                        pcl16::PointXYZ sample_pt, int sample_pt_idx,
                                                                        double sample_spread, double hull_alpha,
                                                                        double hist_res);

cpl_visual_features::ShapeDescriptor extractLocalShapeFeatures(XYZPointCloud& samples_pcl,
                                                               ProtoObject& cur_obj, pcl16::PointXYZ sample_loc,
                                                               double s, double hull_alpha, double hist_res);

cv::Mat computeShapeFeatureAffinityMatrix(ShapeLocations& locs, bool use_center = false);

double shapeFeatureChiSquareDist(cpl_visual_features::ShapeDescriptor& a, cpl_visual_features::ShapeDescriptor& b, double gamma=0.0);

double shapeFeatureSquaredEuclideanDist(cpl_visual_features::ShapeDescriptor& a, cpl_visual_features::ShapeDescriptor& b);

void clusterShapeFeatures(ShapeLocations& locs, int k, std::vector<int>& cluster_ids,
                          cpl_visual_features::ShapeDescriptors& centers, double min_err_change, int max_iter,
                          int num_retries = 5);

int closestShapeFeatureCluster(cpl_visual_features::ShapeDescriptor& descriptor,
                               cpl_visual_features::ShapeDescriptors& centers, double& min_dist);

cpl_visual_features::ShapeDescriptors loadSVRTrainingFeatures(std::string feature_path, int feat_length);

cv::Mat computeChi2Kernel(cpl_visual_features::ShapeDescriptors& sds, std::string feat_path, int local_length,
                          int global_length, double gamma_local, double gamma_global, double mixture_weight);

};
#endif // shape_features_h_DEFINED
