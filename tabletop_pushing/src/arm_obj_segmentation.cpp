#include <tabletop_pushing/arm_obj_segmentation.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tabletop_pushing/extern/graphcut/energy.h>
#include <tabletop_pushing/extern/graphcut/GCoptimization.h>

#include <map>
#include <algorithm>
#include <iostream>

// #define VISUALIZE_GRAPH_WEIGHTS 1
// #define USE_CANNY_EDGES 1

namespace tabletop_pushing
{
class NodeTable
{
 public:
  typedef std::map<int, std::map<int, int> >::iterator TableIt;
  typedef std::map<int, int>::iterator CellIt;
  typedef std::map<int, int> Cell;

  int find(int r, int c)
  {
    TableIt it = table_.find(r);
    if (it == table_.end())
    {
      return -1;
    }
    CellIt cit = table_[r].find(c);
    if (cit == table_[r].end())
    {
      return -1;
    }
    return table_[r][c];
  }

  void addNode(int r, int c, int idx)
  {
    TableIt it = table_.find(r);
    if (it == table_.end())
    {
      Cell cell_r;
      cell_r[c] = idx;
      table_[r] = cell_r;
    }
    else
    {
      table_[r][c] = idx;
    }
  }

  int getIdx(int r, int c)
  {
    return table_[r][c];
  }
 protected:
  std::map<int, std::map<int, int> > table_;
};

ArmObjSegmentation::ArmObjSegmentation(float fg_tied_weight, float bg_tied_weight, float bg_enlarge_size,
                                       float arm_enlarge_width, float arm_shrink_width, float sigma, float lambda) :
    fg_tied_weight_(fg_tied_weight), bg_tied_weight_(bg_tied_weight), bg_enlarge_size_(bg_enlarge_size),
    arm_enlarge_width_(arm_enlarge_width), arm_shrink_width_(arm_shrink_width),
    sigma_d_(sigma), pairwise_lambda_(lambda), have_arm_color_model_(false), have_bg_color_model_(false)
{
  // Create derivative kernels for edge calculation
  cv::getDerivKernels(dy_kernel_, dx_kernel_, 1, 0, CV_SCHARR, true, CV_32F);
  // cv::flip(dy_kernel_, dy_kernel_, -1);
  cv::transpose(dy_kernel_, dx_kernel_);
}


cv::Mat ArmObjSegmentation::segment(cv::Mat& color_img, cv::Mat& depth_img, cv::Mat& self_mask,
                                    cv::Mat& table_mask, bool init_color_models)
{

  cv::Mat color_img_lab_uchar(color_img.size(), color_img.type());
  cv::Mat color_img_lab(color_img.size(), CV_32FC3);
  cv::cvtColor(color_img, color_img_lab_uchar, CV_BGR2HSV);
  // cv::cvtColor(color_img, color_img_lab_uchar, CV_BGR2Lab);
  color_img_lab_uchar.convertTo(color_img_lab, CV_32FC3, 1.0/255);
  cv::Mat tmp_bw(color_img.size(), CV_8UC1);
  cv::Mat bw_img(color_img.size(), CV_32FC1);
  // Convert to grayscale
  cv::cvtColor(color_img, tmp_bw, CV_BGR2GRAY);
  tmp_bw.convertTo(bw_img, CV_32FC1, 1.0/255);


#ifdef VISUALIZE_GRAPH_WEIGHTS
  cv::Mat fg_weights(color_img.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat fg_tied_weights(color_img.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat bg_tied_weights(color_img.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat bg_weights(color_img.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat wu_weights(color_img.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat wl_weights(color_img.size(), CV_32FC1, cv::Scalar(0.0));
#endif // VISUALIZE_GRAPH_WEIGHTS

  // TODO: Clean this up once we have stuff working well
  cv::Mat inv_self_mask;
  cv::bitwise_not(self_mask, inv_self_mask);
  cv::Mat much_larger_mask(self_mask.size(), CV_8UC1, cv::Scalar(255));
  cv::Mat enlarge_element(bg_enlarge_size_, bg_enlarge_size_, CV_8UC1, cv::Scalar(255));
  cv::dilate(inv_self_mask, much_larger_mask, enlarge_element);
  cv::Mat larger_mask, known_arm_mask;
  cv::Mat arm_band = getArmBand(inv_self_mask, arm_enlarge_width_, arm_shrink_width_, false,
                                larger_mask, known_arm_mask);

  // Get known arm pixels
  cv::Mat known_arm_pixels;
  color_img_lab.copyTo(known_arm_pixels, known_arm_mask);
  cv::Mat known_bg_mask = much_larger_mask - larger_mask;

  // Get known object pixels
  cv::Mat known_bg_pixels;
  color_img_lab.copyTo(known_bg_pixels, known_bg_mask);

  // Get stats for building graph
  int num_nodes = 0;
  int num_edges = 0;

  tabletop_pushing::NodeTable nt;
  for (int r = 0; r < much_larger_mask.rows; ++r)
  {
    for (int c = 0; c < much_larger_mask.cols; ++c)
    {
      if (much_larger_mask.at<uchar>(r,c) > 0)
      {
        // Add this as another node
        int cur_idx = num_nodes++;
        nt.addNode(r, c, cur_idx);
        int test_idx = nt.getIdx(r,c);
        // Check for edges to add
        // left edge
        if (c > 0 && much_larger_mask.at<uchar>(r, c-1) > 0)
        {
          num_edges++;
        }
        if (r > 0)
        {
          // Up edge
          if(much_larger_mask.at<uchar>(r-1,c) > 0)
          {
            num_edges++;
          }
        }
      }
    }
  }
  if (num_nodes < 1)
  {
    cv::Mat empty(color_img.size(), CV_8UC1, cv::Scalar(0));
    return empty;
  }

  if (!have_arm_color_model_)
  {
    arm_color_model_ = getGMMColorModel(known_arm_pixels, known_arm_mask, 3);
    arm_color_model_.dispparams();
    have_arm_color_model_ = true;
  }
  if(init_color_models || !have_bg_color_model_)
  {
    bg_color_model_ = getGMMColorModel(known_bg_pixels, known_bg_mask, 5);
    bg_color_model_.dispparams();
    have_bg_color_model_ = true;
  }

#ifdef USE_CANNY_EDGES
  cv::Mat canny_edges_8bit;
  cv::Mat canny_edges;
  cv::Canny(tmp_bw, canny_edges_8bit, 120, 250);
  canny_edges_8bit.convertTo(canny_edges, CV_32FC1, 1.0/255);
#else
  cv::Mat Ix = getXImageDeriv(bw_img);
  cv::Mat Iy = getYImageDeriv(bw_img);
#endif
  cv::Mat Dx = getXImageDeriv(depth_img);
  cv::Mat Dy = getYImageDeriv(depth_img);
  tabletop_pushing::GraphType *g;
  g = new GraphType(num_nodes, num_edges);
  // Populate unary and binary edge weights
  for (int r = 0; r < much_larger_mask.rows; ++r)
  {
    for (int c = 0; c < much_larger_mask.cols; ++c)
    {
      if (much_larger_mask.at<uchar>(r,c) > 0)
      {
        int cur_idx = nt.getIdx(r,c);
        float fg_weight = 0.0;
        float bg_weight = 0.0;
        if (known_arm_mask.at<uchar>(r,c) > 0)
        {
          fg_weight = fg_tied_weight_;
          bg_weight = 0.0;
        }
        else if (known_bg_mask.at<uchar>(r,c) > 0 || table_mask.at<uchar>(r,c) > 0)
        {
          fg_weight = 0.0;
          bg_weight = bg_tied_weight_;
        }
        else
        {
          fg_weight = getUnaryWeight(color_img_lab.at<cv::Vec3f>(r,c), arm_color_model_);
          bg_weight = getUnaryWeight(color_img_lab.at<cv::Vec3f>(r,c), bg_color_model_);
        }
        g->add_node();
        g->add_tweights(cur_idx, fg_weight, bg_weight);

#ifdef VISUALIZE_GRAPH_WEIGHTS
        fg_weights.at<float>(r,c) = fg_weight;
        bg_weights.at<float>(r,c) = bg_weight;
#endif // VISUALIZE_GRAPH_WEIGHTS

        // Add left link
        if (c > 0 && much_larger_mask.at<uchar>(r, c-1) > 0)
        {
          int other_idx = nt.getIdx(r, c-1);
#ifdef USE_CANNY_EDGES
          float w_l = getEdgeWeightBoundary(canny_edges.at<float>(r,c), Dx.at<float>(r,c),
                                            canny_edges.at<float>(r, c-1), Dx.at<float>(r, c-1));
#else // USE_CANNY_EDGES
          float w_l = getEdgeWeightBoundary(Ix.at<float>(r,c), Dx.at<float>(r,c),
                                            Ix.at<float>(r, c-1), Dx.at<float>(r, c-1));
#endif // USE_CANNY_EDGES
          g->add_edge(cur_idx, other_idx, /*capacities*/ w_l, w_l);

#ifdef VISUALIZE_GRAPH_WEIGHTS
          wl_weights.at<float>(r,c) = w_l;
#endif // VISUALIZE_GRAPH_WEIGHTS
        }
        if (r > 0)
        {
          // Add up link
          if(much_larger_mask.at<uchar>(r-1,c) > 0)
          {
            int other_idx = nt.getIdx(r-1, c);
#ifdef USE_CANNY_EDGES
            float w_u = getEdgeWeightBoundary(canny_edges.at<float>(r, c),   Dy.at<float>(r, c),
                                              canny_edges.at<float>(r-1, c), Dy.at<float>(r-1, c));
#else // USE_CANNY_EDGES
            float w_u = getEdgeWeightBoundary(Iy.at<float>(r,c), Dx.at<float>(r,c),
                                              Iy.at<float>(r, c-1), Dx.at<float>(r, c-1));
#endif // USE_CANNY_EDGES
            g->add_edge(cur_idx, other_idx, /*capacities*/ w_u, w_u);

#ifdef VISUALIZE_GRAPH_WEIGHTS
            wu_weights.at<float>(r,c) = w_u;
#endif // VISUALIZE_GRAPH_WEIGHTS
          }
        }
      }
    }
  }

  // Perform cut
  g->maxflow(false);
  // Convert output into image
  cv::Mat segs = convertFlowToMat(g, nt, color_img.rows, color_img.cols);
  // Cleanup
  delete g;

  cv::Mat graph_input;
  cv::Mat segment_arm;
  cv::Mat segment_bg;
  color_img.copyTo(graph_input, much_larger_mask);
  color_img.copyTo(segment_arm, segs);
  color_img.copyTo(segment_bg, (segs == 0));
  //cv::imshow("segments", segs);
  // cv::imshow("Graph input", graph_input);
  cv::imshow("Arm segment", segment_arm);
  // cv::imshow("Background segment", segment_bg);
  cv::imshow("Table mask", table_mask);
  // cv::imshow("Known bg mask", known_bg_mask);

#ifdef VISUALIZE_GRAPH_WEIGHTS
  double min_val, max_val;
  cv::minMaxLoc(fg_weights, &min_val, &max_val);
  cv::imshow("fg weights", (fg_weights-min_val)/(max_val-min_val));
  std::cout << "Max fg weight: " << max_val << std::endl;
  std::cout << "Min fg weight: " << min_val << std::endl;

  cv::minMaxLoc(bg_weights, &min_val, &max_val);
  cv::imshow("bg weights", (bg_weights-min_val)/(max_val-min_val));
  std::cout << "Max bg weight: " << max_val << std::endl;
  std::cout << "Min bg weight: " << min_val << std::endl;

  cv::minMaxLoc(wu_weights, &min_val, &max_val);
  cv::imshow("up weights", wu_weights/max_val);
  std::cout << "Max up weight: " << max_val << std::endl;
  std::cout << "Min up weight: " << min_val << std::endl;

  cv::minMaxLoc(wl_weights, &min_val, &max_val);
  cv::imshow("left weights", wl_weights/max_val);
  std::cout << "Max left weight: " << max_val << std::endl;
  std::cout << "Min left weight: " << min_val << std::endl;
#endif // VISUALIZE_GRAPH_WEIGHTS
#ifdef USE_CANNY_EDGES
  cv::imshow("Canny", canny_edges);
#endif
  return segs;
}

//
// Helper Methods
//

cv::Mat ArmObjSegmentation::getMorphCross(int img_size, int cross_width)
{
  cv::Mat morph_element(img_size, img_size, CV_8UC1, cv::Scalar(0));
  int img_d = img_size / 2;
  int cross_d = std::max(cross_width/2,1);
  for (int r = 0; r < morph_element.rows; ++r)
  {
    for (int c = 0; c < morph_element.cols; ++c)
    {
      if (abs(img_d - r) < cross_d || abs(img_d - c) < cross_d)
      {
        morph_element.at<uchar>(r,c) = 255;
      }
    }
  }
  return morph_element;
}

cv::Mat ArmObjSegmentation::convertFlowToMat(GraphType *g, NodeTable& nt, int R, int C)
{
  cv::Mat segs(R,C, CV_8UC1, cv::Scalar(0));
  for (int r = 0; r < R; ++r)
  {
    uchar* seg_row = segs.ptr<uchar>(r);
    for (int c = 0; c < C; ++c)
    {
      int idx = nt.find(r,c);
      if (idx < 0) continue;
      int label = (g->what_segment(idx) == GraphType::SOURCE);
      seg_row[c] = label*255;
    }
  }
  return segs;
}

float ArmObjSegmentation::getEdgeWeightBoundary(float I0, float d0, float I1, float d1)
{
  // float w = pairwise_lambda_*exp(-std::max(fabs(I0)+fabs(d0), fabs(I1)+fabs(d1))/sigma_d_);
  // float w = pairwise_lambda_*exp(-std::max(fabs(I0), fabs(I1)));
  float w = pairwise_lambda_*exp(-fabs(I0));
  return w;
}

cv::Mat ArmObjSegmentation::getXImageDeriv(cv::Mat& input_img)
{
  cv::Mat Ix(input_img.size(), CV_32FC1);
  // Get image X derivative
  cv::filter2D(input_img, Ix, CV_32F, dx_kernel_);
  return Ix;
}

cv::Mat ArmObjSegmentation::getYImageDeriv(cv::Mat& input_img)
{
  cv::Mat Iy(input_img.size(), CV_32FC1);
  // Get image Y derivative
  cv::filter2D(input_img, Iy, CV_32F, dy_kernel_);
  return Iy;
}

cv::Mat ArmObjSegmentation::getArmBand(cv::Mat& input_mask, int enlarge_width,
                                       int shrink_width, bool input_inverted)
{
  cv::Mat larger_mask, smaller_mask;
  return getArmBand(input_mask, enlarge_width, shrink_width, input_inverted, larger_mask, smaller_mask);
}

/**
 * Method to create a mask near the perimeter of the robots projected arm mask
 *
 * @param input_mask The robot's projected arm mask
 * @param enlarge_width Amount to dilate the arm mask to get outer band
 * @param shrink_width Amount to erode the arm mask to get the innner band
 * @param input_inverted Set to true if arm is 0 and background is 255
 *
 * @return A binary image of the band near the edges of the robot arm
 */
cv::Mat ArmObjSegmentation::getArmBand(cv::Mat& input_mask, int enlarge_width,
                                       int shrink_width, bool input_inverted,
                                       cv::Mat& larger_mask,  cv::Mat& smaller_mask)
{
  cv::Mat mask;
  if (input_inverted) // Swap if the arm is negative and the background positive
  {
    cv::bitwise_not(input_mask, mask);
  }
  else
  {
    mask = input_mask;
  }

  cv::Mat enlarge_element(enlarge_width, enlarge_width, CV_8UC1, cv::Scalar(255));
  cv::Mat shrink_element(shrink_width, shrink_width, CV_8UC1, cv::Scalar(255));
  cv::dilate(mask, larger_mask, enlarge_element);
  cv::erode(mask, smaller_mask, shrink_element);
  cv::Mat arm_band = larger_mask - smaller_mask;
  return arm_band;
}

GMM ArmObjSegmentation::getGMMColorModel(cv::Mat& samples, cv::Mat& mask, int nc)
{
  std::vector<cv::Vec3f> pnts;
  for (int r = 0; r < samples.rows; ++r)
  {
    for (int c = 0; c < samples.cols; ++c)
    {
      if (mask.at<uchar>(r,c) > 0)
      {
        pnts.push_back(samples.at<cv::Vec3f>(r,c));
      }
    }
  }
  GMM color_model(0.0001);
  color_model.alloc(nc);
  if (pnts.size() > 1)
  {
    color_model.kmeansInit(pnts, 0.05);
    color_model.GmmEm(pnts);
    color_model.dispparams();
  }
  return color_model;
}

float ArmObjSegmentation::getUnaryWeight(cv::Vec3f sample, GMM& color_model)
{
  return color_model.grabCutLikelihood(sample);
}

void ArmObjSegmentation::loadArmColorModel(std::string file_path)
{
  arm_color_model_.loadgmm(fs::path(file_path));
  have_arm_color_model_ = true;
}

void ArmObjSegmentation::loadBGColorModel(std::string file_path)
{
  bg_color_model_.loadgmm(fs::path(file_path));
  have_bg_color_model_ = true;
}

void ArmObjSegmentation::setBGColorModel(GMM& new_bg_model)
{
  bg_color_model_ = new_bg_model;
  have_bg_color_model_ = true;
}

}; // namespace tabletop_pushing
