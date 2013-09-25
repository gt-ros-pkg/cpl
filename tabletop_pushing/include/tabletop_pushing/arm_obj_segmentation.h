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
  ArmObjSegmentation(float fg_tied_weight = 10.0, float bg_tied_weight = 15.0, float bg_enlarge_size = 100,
                     float arm_enlarge_width = 15, float arm_shrink_width_ = 15,float sigma = 1.0, float lambda = 5.0);
  cv::Mat segment(cv::Mat& color_img, cv::Mat& depth_img, cv::Mat& self_mask, cv::Mat& table_mask,
                  bool init_color_models=true);
  static cv::Mat getArmBand(cv::Mat& input_mask, int enlarge_width, int shrink_width, bool input_inverted=true);
  static cv::Mat getArmBand(cv::Mat& input_mask, int enlarge_width, int shrink_width, bool input_inverted,
                            cv::Mat& larger_mask,  cv::Mat& smaller_mask);
  void loadArmColorModel(std::string file_path);
  void loadBGColorModel(std::string file_path);
  void setBGColorModel(GMM& new_bg_model);
  void buildBGColorModel(GMM& table_color_model, GMM& obj_color_model);
 protected:
  cv::Mat getXImageDeriv(cv::Mat& color_img);
  cv::Mat getYImageDeriv(cv::Mat& color_img);
  static cv::Mat getMorphCross(int img_size, int cross_width);
  static cv::Mat convertFlowToMat(tabletop_pushing::GraphType *g, tabletop_pushing::NodeTable& nt, int R, int C);

  static float getUnaryWeight(cv::Vec3f sample, GMM& fg_color_model);
  static GMM getGMMColorModel(cv::Mat& samples, cv::Mat& mask, int nc=10);

  float getEdgeWeightBoundary(float c0, float d0, float c1, float d1);

  // Members
  float fg_tied_weight_;
  float bg_tied_weight_;
  float bg_enlarge_size_;
  float arm_enlarge_width_;
  float arm_shrink_width_;
  float sigma_d_;
  float pairwise_lambda_;
  cv::Mat dy_kernel_;
  cv::Mat dx_kernel_;
  GMM arm_color_model_;
  GMM bg_color_model_;
  bool have_arm_color_model_;
  bool have_bg_color_model_;
};
};
#endif // arm_obj_segmentation_h_DEFINED
