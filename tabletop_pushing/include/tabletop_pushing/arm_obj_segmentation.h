// OpenCV
#include <opencv2/core/core.hpp>
#include <tabletop_pushing/extern/graphcut/graph.h>

#ifndef arm_obj_segmentation_h_DEFINED
#define arm_obj_segmentation_h_DEFINED 1
namespace tabletop_pushing
{

typedef Graph<float, float, float> GraphType;
class NodeTable;

class ArmObjSegmentation
{
 public:
  static void segment(cv::Mat& color_img, cv::Mat& depth_img, cv::Mat& self_mask);
  static void getArmEdges(cv::Mat& color_img, cv::Mat& depth_img, cv::Mat& self_mask);
  static cv::Mat getArmBand(cv::Mat& input_mask, int enlarge_width, int shrink_width, bool input_inverted=true);
  static cv::Mat getArmBand(cv::Mat& input_mask, int enlarge_width, int shrink_width, bool input_inverted,
                            cv::Mat& larger_mask,  cv::Mat& smaller_mask);
  // TODO: Make non-static using parameters;
  static float getEdgeWeight(cv::Vec3f c0, float d0, cv::Vec3f c1, float d1);

 protected:
  static cv::Mat convertFlowToMat(tabletop_pushing::GraphType *g, tabletop_pushing::NodeTable& nt, int R, int C);
  static float getUnaryWeight(cv::Vec3f sample, cv::Vec3f mean, cv::Vec3f var);
  static void getColorModel(cv::Mat& samples, cv::Vec3f& mean, cv::Vec3f& var, cv::Mat& mask);

  float w_c_alpha_;
  float w_c_beta_;
  float w_c_gamma_;
  float w_c_eta_;
};
};
#endif // arm_obj_segmentation_h_DEFINED
