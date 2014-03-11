#include <libsvm/svm.h>
#include <sstream>
#include <string>
#include <ros/ros.h>
#include <tabletop_pushing/shape_features.h>

using cpl_visual_features::ShapeDescriptors;
using cpl_visual_features::ShapeDescriptor;

int main(int argc, char** argv)
{
  std::string param_path;
  param_path = "/u/thermans/src/gt-ros-pkg/cpl/tabletop_pushing/cfg/push_svm_ichr_no_food_box.model";
  ROS_INFO_STREAM("param_path " << param_path);
  svm_model* push_model;
  push_model = svm_load_model(param_path.c_str());
  ROS_INFO_STREAM("svm type: " << svm_get_svm_type(push_model));
  ROS_INFO_STREAM("nr classes: " << svm_get_nr_class(push_model));
  // int num_sv = svm_get_nr_sv(push_model);
  // ROS_INFO_STREAM("nr sv: " << num_sv);
  ROS_INFO_STREAM("rho: " << *push_model->rho);
  svm_parameter params = push_model->param;
  ROS_INFO_STREAM("kernel type: " << params.kernel_type);
  // ROS_INFO_STREAM("free_sv: " << push_model->free_sv);
  // std::stringstream sv_idx;
  // std::stringstream sv_coef;
  // for (int i = 0; i < num_sv; ++i)
  // {
  //   sv_idx << " " << push_model->SV[i]->value;
  //   sv_idx << ":" << push_model->sv_coef[push_model->SV[i]->index][i];
  // }
  // ROS_INFO_STREAM("SVs: " << sv_idx.str());


  int local_length = 36;
  int global_length = 60;
  std::string test_feat_path = "/u/thermans/src/gt-ros-pkg/cpl/tabletop_pushing/cfg/push_svm_ichr_no_food_box-feats.txt";
  ShapeDescriptors sds = tabletop_pushing::loadSVRTrainingFeatures(test_feat_path, local_length+global_length);

  // Remove trailing .model
  param_path.erase(param_path.size()-6, 6);
  std::stringstream train_feat_path;
  train_feat_path << param_path << "-feats.txt";
  ROS_INFO_STREAM(train_feat_path.str());

  double gamma_local = 2.5;
  double gamma_global = 2.0;
  double mixture_weight = 0.7;

  cv::Mat K = tabletop_pushing::computeChi2Kernel(sds, train_feat_path.str(), local_length, global_length,
                                                  gamma_local, gamma_global, mixture_weight);
  // cv::imshow("K matrix", K);
  // cv::waitKey();

  ROS_INFO_STREAM("size(K): (" << K.rows << ", " << K.cols << ")");

  // TODO: Compare K here to MATLAB K
  std::stringstream scores_out;
  std::vector<double> pred_push_scores;
  for (int i = 0; i < K.cols; ++i)
  {
    svm_node* x = new svm_node[K.rows+1];
    x[0].value = 0;
    x[0].index = 0;
    for (int j = 0; j < K.rows; ++j)
    {
      // NOTE: Need to add 1 to index since support vectors are 1 based indices (thanks MATLAB!)
      x[j+1].value = K.at<double>(j, i);
      x[j+1].index = 0; // unused
    }
    // Perform prediction and convert out of log space
    // TODO: Collapse below once we get the bugs worked out
    double raw_pred_score = svm_predict(push_model, x);
    delete x;
    pred_push_scores.push_back(raw_pred_score);
    scores_out  << raw_pred_score << "    ";
  }
  ROS_INFO_STREAM("\n" << scores_out.str());

  return 0;
}
