// OpenCV
#include <opencv2/core/core.hpp>
#include <tabletop_pushing/extern/graphcut/graph.h>
#include <tabletop_pushing/extern/gmm/gmm.h>

#ifndef arm_obj_segmentation_h_DEFINED
#define arm_obj_segmentation_h_DEFINED 1
namespace tabletop_pushing
{

typedef Graph<float, float, float> GraphType;
class NodeTable;

class ArmObjSegmentation
{
 public:
  static cv::Mat segment(cv::Mat& color_img, cv::Mat& depth_img, cv::Mat& self_mask, cv::Mat& table_mask);
  static void getArmEdges(cv::Mat& color_img, cv::Mat& depth_img, cv::Mat& self_mask);
  static cv::Mat getArmBand(cv::Mat& input_mask, int enlarge_width, int shrink_width, bool input_inverted=true);
  static cv::Mat getArmBand(cv::Mat& input_mask, int enlarge_width, int shrink_width, bool input_inverted,
                            cv::Mat& larger_mask,  cv::Mat& smaller_mask);
  // TODO: Make non-static using parameters;
 protected:
  static cv::Mat getXImageDeriv(cv::Mat& color_img);
  static cv::Mat getYImageDeriv(cv::Mat& color_img);
  static cv::Mat getMorphCross(int img_size, int cross_width);
  static cv::Mat convertFlowToMat(tabletop_pushing::GraphType *g, tabletop_pushing::NodeTable& nt, int R, int C);

  static float getUnaryWeight(cv::Vec3f sample, GMM& fg_color_model);
  static GMM getGMMColorModel(cv::Mat& samples, cv::Mat& mask, int nc=10);
  static float getEdgeWeightBoundary(float c0, float d0, float c1, float d1);
};
};
#endif // arm_obj_segmentation_h_DEFINED
