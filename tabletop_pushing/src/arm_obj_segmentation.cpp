#include <tabletop_pushing/arm_obj_segmentation.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tabletop_pushing/extern/graphcut/energy.h>
#include <tabletop_pushing/extern/graphcut/GCoptimization.h>

#include <map>
#include <algorithm>

#define VISUALIZE_GRAPH_WEIGHTS 1
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

cv::Mat ArmObjSegmentation::segment(cv::Mat& color_img, cv::Mat& depth_img, cv::Mat& self_mask)
{
  cv::Mat color_frame_hsv(color_img.size(), color_img.type());
  cv::cvtColor(color_img, color_frame_hsv, CV_BGR2HSV);
  cv::Mat color_frame_f(color_frame_hsv.size(), CV_32FC3);
  color_frame_hsv.convertTo(color_frame_f, CV_32FC3, 1.0/255, 0);
  cv::Mat tmp_bw(color_img.size(), CV_8UC1);
  cv::Mat bw_img(color_img.size(), CV_32FC1);
  // Convert to grayscale
  cv::cvtColor(color_img, tmp_bw, CV_BGR2GRAY);
  tmp_bw.convertTo(bw_img, CV_32FC1, 1.0/255);

#ifdef VISUALIZE_GRAPH_WEIGHTS
  cv::Mat fg_weights(color_frame_f.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat bg_weights(color_frame_f.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat wu_weights(color_frame_f.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat wl_weights(color_frame_f.size(), CV_32FC1, cv::Scalar(0.0));
  // cv::Mat wul_weights(color_frame_f.size(), CV_32FC1, cv::Scalar(0.0));
#endif // VISUALIZE_GRAPH_WEIGHTS

  // TODO: Clean this up once we have stuff working well
  cv::Mat inv_self_mask;
  cv::bitwise_not(self_mask, inv_self_mask);
  cv::Mat much_larger_mask(self_mask.size(), CV_8UC1, 255);
  // cv::Mat enlarge_element(50, 50, CV_8UC1, cv::Scalar(255));
  // cv::dilate(inv_self_mask, much_larger_mask, enlarge_element);
  cv::Mat larger_mask, known_arm_mask;
  cv::Mat arm_band = getArmBand(inv_self_mask, 15, 15, false, larger_mask, known_arm_mask);

  // Get known arm pixels
  cv::Mat known_arm_pixels;
  color_frame_f.copyTo(known_arm_pixels, known_arm_mask);
  cv::Mat known_bg_mask = much_larger_mask - larger_mask;

  // Get known object pixels
  cv::Mat known_bg_pixels;
  color_frame_f.copyTo(known_bg_pixels, known_bg_mask);

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
          // // Up-left edge
          // if(c > 0 && much_larger_mask.at<uchar>(r-1,c-1) > 0)
          // {
          //   num_edges++;
          // }
        }
      }
    }
  }
  if (num_nodes < 1)
  {
    cv::Mat empty(color_img.size(), CV_8UC1, cv::Scalar(0));
    return empty;
  }

  // Build color models
  cv::Vec3f fg_mean;
  cv::Vec3f fg_var;
  getColorModel(known_arm_pixels, fg_mean, fg_var, known_arm_mask);

  cv::Vec3f bg_mean;
  cv::Vec3f bg_var;
  getColorModel(known_bg_pixels, bg_mean, bg_var, known_bg_mask);

  cv::Mat Ix = getXImageDeriv(bw_img);
  cv::Mat Iy = getYImageDeriv(bw_img);
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
        float fg_weight = 0.5;
        float bg_weight = 0.5;
        if (known_arm_mask.at<uchar>(r,c) > 0)
        {
          fg_weight = 2.0;
          bg_weight = 0.0;
        }
        else if (known_bg_mask.at<uchar>(r,c) > 0)
        {
          fg_weight = 0.0;
          bg_weight = 2.0;
        }
        else
        {
          fg_weight = getUnaryWeight(color_frame_f.at<cv::Vec3f>(r,c), fg_mean, fg_var, bg_mean, bg_var);
          bg_weight = 1.0 - fg_weight;
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
          // float w_l = getEdgeWeight(color_frame_f.at<cv::Vec3f>(r, c),   depth_img.at<float>(r, c),
          //                           color_frame_f.at<cv::Vec3f>(r, c-1), depth_img.at<float>(r, c-1));
          float w_l = fabs(Ix.at<float>(r,c)) + fabs(Dx.at<float>(r,c));
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
            // float w_u = getEdgeWeight(color_frame_f.at<cv::Vec3f>(r, c),   depth_img.at<float>(r, c),
            //                           color_frame_f.at<cv::Vec3f>(r-1, c), depth_img.at<float>(r-1, c));
            float w_u = fabs(Iy.at<float>(r,c)) + fabs(Dy.at<float>(r,c));
            g->add_edge(cur_idx, other_idx, /*capacities*/ w_u, w_u);

#ifdef VISUALIZE_GRAPH_WEIGHTS
            wu_weights.at<float>(r,c) = w_u;
#endif // VISUALIZE_GRAPH_WEIGHTS
          }
          // Add up-left link
//           if(c > 0 && much_larger_mask.at<uchar>(r-1,c-1) > 0)
//           {
//             int other_idx = nt.getIdx(r-1, c-1);
//             float w_ul = fabs(Ixy.at<float>(r,c)) + fabs(Dxy.at<float>(r,c));
//             // float w_ul = getEdgeWeight(color_frame_f.at<cv::Vec3f>(r, c),     depth_img.at<float>(r, c),
//             //                            color_frame_f.at<cv::Vec3f>(r-1, c-1), depth_img.at<float>(r-1, c-1));
//              g->add_edge(cur_idx, other_idx, /*capacities*/ w_ul, w_ul);

// #ifdef VISUALIZE_GRAPH_WEIGHTS
//            wul_weights.at<float>(r,c) = w_ul;
// #endif // VISUALIZE_GRAPH_WEIGHTS
//            }
        }
      }
    }
  }

#ifdef VISUALIZE_GRAPH_WEIGHTS
  cv::imshow("up weights", wu_weights);
  cv::imshow("left weights", wl_weights);
  // cv::imshow("up-left weights", wul_weights);
  cv::imshow("fg weights", fg_weights);
  cv::imshow("bg weights", bg_weights);
#endif // VISUALIZE_GRAPH_WEIGHTS

  // Perform cut
  g->maxflow(false);

  // TODO: Convert output into image
  cv::Mat segs = convertFlowToMat(g, nt, color_frame_f.rows, color_frame_f.cols);
  // Cleanup
  delete g;

  // TODO: Make method for generating a morph cross of different sizes (if abs(r-(width/2)) < cross_width or c...
  cv::Mat segs_cleaned;
  cv::Mat open_element(3, 3, CV_8UC1, cv::Scalar(255));
  cv::erode(segs, segs_cleaned, open_element);
  cv::dilate(segs_cleaned, segs_cleaned, open_element);

  cv::Mat graph_input;
  cv::Mat segment_arm;
  cv::Mat segment_bg;
  color_img.copyTo(graph_input, much_larger_mask);
  color_img.copyTo(segment_arm, segs_cleaned);
  color_img.copyTo(segment_bg, (segs_cleaned == 0));
  cv::imshow("segments", segs_cleaned);
  cv::imshow("Graph input", graph_input);
  cv::imshow("Segment bg", segment_bg);
  cv::imshow("Segment arm", segment_arm);
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

float ArmObjSegmentation::getEdgeWeight(cv::Vec3f c0, float d0, cv::Vec3f c1, float d1)
{
  float w_c_alpha_ = 0.5;
  float w_c_beta_ = 0.5;
  float w_c_gamma_ = 0.0;
  float w_c_eta_ = 0.0;
  cv::Vec3f c_d = c0-c1;
  float w_d = d0-d1;
  return (w_c_alpha_*(exp(std::abs(c_d[0]))-1.0) + w_c_beta_*(exp(std::abs(c_d[1]))-1.0) +
           w_c_beta_*(exp(std::abs(c_d[2]))-1.0) +  w_c_eta_*(exp(std::abs(w_d))-1.0));
}

void ArmObjSegmentation::getArmEdges(cv::Mat& color_img, cv::Mat& depth_img, cv::Mat& self_mask)
{
  // Speed up later
  cv::Mat dy_kernel;
  cv::Mat dx_kernel;
  // Create derivative kernels for edge calculation
  cv::getDerivKernels(dy_kernel, dx_kernel, 1, 0, CV_SCHARR, true, CV_32F);
  // cv::flip(dy_kernel, dy_kernel, -1);
  cv::transpose(dy_kernel, dx_kernel);
  cv::Mat tmp_bw(color_img.size(), CV_8UC1);
  cv::Mat bw_img(color_img.size(), CV_32FC1);
  cv::Mat Ix(bw_img.size(), CV_32FC1);
  cv::Mat Iy(bw_img.size(), CV_32FC1);
  cv::Mat Ix_d(bw_img.size(), CV_32FC1);
  cv::Mat Iy_d(bw_img.size(), CV_32FC1);
  cv::Mat edge_img(color_img.size(), CV_32FC1);
  cv::Mat depth_edge_img(color_img.size(), CV_32FC1);
  cv::Mat edge_img_masked(edge_img.size(), CV_32FC1, cv::Scalar(0.0));
  cv::Mat depth_edge_img_masked(edge_img.size(), CV_32FC1, cv::Scalar(0.0));

  // Convert to grayscale
  cv::cvtColor(color_img, tmp_bw, CV_BGR2GRAY);
  tmp_bw.convertTo(bw_img, CV_32FC1, 1.0/255);

  // Get image derivatives
  cv::filter2D(bw_img, Ix, CV_32F, dx_kernel);
  cv::filter2D(bw_img, Iy, CV_32F, dy_kernel);
  cv::filter2D(depth_img, Ix_d, CV_32F, dx_kernel);
  cv::filter2D(depth_img, Iy_d, CV_32F, dy_kernel);

  // Create magintude image
  for (int r = 0; r < edge_img.rows; ++r)
  {
    float* mag_row = edge_img.ptr<float>(r);
    float* Ix_row = Ix.ptr<float>(r);
    float* Iy_row = Iy.ptr<float>(r);
    for (int c = 0; c < edge_img.cols; ++c)
    {
      mag_row[c] = sqrt(Ix_row[c]*Ix_row[c] + Iy_row[c]*Iy_row[c]);
    }
  }
  for (int r = 0; r < depth_edge_img.rows; ++r)
  {
    float* mag_row = depth_edge_img.ptr<float>(r);
    float* Ix_row = Ix_d.ptr<float>(r);
    float* Iy_row = Iy_d.ptr<float>(r);
    for (int c = 0; c < depth_edge_img.cols; ++c)
    {
      mag_row[c] = sqrt(Ix_row[c]*Ix_row[c] + Iy_row[c]*Iy_row[c]);
    }
  }

  cv::Mat arm_band = getArmBand(self_mask, 15, 15, true);
  edge_img.copyTo(edge_img_masked, arm_band);
  depth_edge_img.copyTo(depth_edge_img_masked, arm_band);

  cv::imshow("Edge Image", edge_img);
  cv::imshow("Arm edge Image", edge_img_masked);
  cv::imshow("Depth Edge Image", depth_edge_img);
  cv::imshow("Depth Arm edge Image", depth_edge_img_masked);
  cv::imshow("Arm band", arm_band);
}

cv::Mat ArmObjSegmentation::getEdgeImage(cv::Mat& color_img)
{
  // TODO: Move filters into constructor
  // Speed up later
  cv::Mat dy_kernel;
  cv::Mat dx_kernel;
  // Create derivative kernels for edge calculation
  cv::getDerivKernels(dy_kernel, dx_kernel, 1, 0, CV_SCHARR, true, CV_32F);
  // cv::flip(dy_kernel, dy_kernel, -1);
  cv::transpose(dy_kernel, dx_kernel);
  cv::Mat tmp_bw(color_img.size(), CV_8UC1);
  cv::Mat bw_img(color_img.size(), CV_32FC1);
  cv::Mat Ix(bw_img.size(), CV_32FC1);
  cv::Mat Iy(bw_img.size(), CV_32FC1);
  cv::Mat edge_img(color_img.size(), CV_32FC1);

  // Convert to grayscale
  cv::cvtColor(color_img, tmp_bw, CV_BGR2GRAY);
  tmp_bw.convertTo(bw_img, CV_32FC1, 1.0/255);

  // Get image derivatives
  cv::filter2D(bw_img, Ix, CV_32F, dx_kernel);
  cv::filter2D(bw_img, Iy, CV_32F, dy_kernel);

  // Create magintude image
  for (int r = 0; r < edge_img.rows; ++r)
  {
    float* mag_row = edge_img.ptr<float>(r);
    float* Ix_row = Ix.ptr<float>(r);
    float* Iy_row = Iy.ptr<float>(r);
    for (int c = 0; c < edge_img.cols; ++c)
    {
      mag_row[c] = sqrt(Ix_row[c]*Ix_row[c] + Iy_row[c]*Iy_row[c]);
    }
  }
  return edge_img;
}

cv::Mat ArmObjSegmentation::getXImageDeriv(cv::Mat& input_img)
{
  // TODO: Move filters into constructor
  // Speed up later
  cv::Mat dy_kernel;
  cv::Mat dx_kernel;
  // Create derivative kernels for edge calculation
  cv::getDerivKernels(dy_kernel, dx_kernel, 1, 0, CV_SCHARR, true, CV_32F);
  // cv::flip(dy_kernel, dy_kernel, -1);
  cv::transpose(dy_kernel, dx_kernel);
  cv::Mat Ix(input_img.size(), CV_32FC1);
  // Get image derivatives
  cv::filter2D(input_img, Ix, CV_32F, dx_kernel);
  return Ix;
}

cv::Mat ArmObjSegmentation::getYImageDeriv(cv::Mat& input_img)
{
  // TODO: Move filters into constructor
  // Speed up later
  cv::Mat dy_kernel;
  cv::Mat dx_kernel;
  // Create derivative kernels for edge calculation
  cv::getDerivKernels(dy_kernel, dx_kernel, 1, 0, CV_SCHARR, true, CV_32F);
  cv::Mat Iy(input_img.size(), CV_32FC1);
  // Get image derivatives
  cv::filter2D(input_img, Iy, CV_32F, dy_kernel);
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

void ArmObjSegmentation::getColorModel(cv::Mat& samples, cv::Vec3f& mean, cv::Vec3f& var, cv::Mat& mask)
{
  mean[0] = 0.0;
  mean[1] = 0.0;
  mean[2] = 0.0;
  int num_samples = 0;
  for (int r = 0; r < samples.rows; ++r)
  {
    for (int c = 0; c < samples.cols; ++c)
    {
      if (mask.at<uchar>(r,c) > 0)
      {
        mean += samples.at<cv::Vec3f>(r,c);
        num_samples += 1;
      }
    }
  }
  if (num_samples > 0)
  {
    mean[0] /= num_samples;
    mean[1] /= num_samples;
    mean[2] /= num_samples;
  }

  var[0] = 0.0;
  var[1] = 0.0;
  var[2] = 0.0;
  for (int r = 0; r < samples.rows; ++r)
  {
    for (int c = 0; c < samples.cols; ++c)
    {
      if (mask.at<uchar>(r,c) > 0)
      {
        cv::Vec3f diff = samples.at<cv::Vec3f>(r,c) - mean;
        diff = diff.mul(diff);
        var += diff;
      }
    }
  }
  var[0] /= (num_samples+1.0);
  var[1] /= (num_samples+1.0);
  var[2] /= (num_samples+1.0);


  // cv::Mat mean_img(samples.size(), CV_32FC3, cv::Scalar(mean[0], mean[1], mean[2]));
  // cv::cvtColor(mean_img, mean_img, CV_HSV2BGR);
  // cv::imshow("mean_color", mean_img);
}

float ArmObjSegmentation::getUnaryWeight(cv::Vec3f sample, cv::Vec3f mean, cv::Vec3f var)
{
  cv::Vec3f diff = sample - mean;
  // const float h_score = 1/sqrt(2.0*M_PI*var[0])*exp(-fabs(diff[0]*diff[0])/(2.0*var[0]));
  // const float s_score = 1/sqrt(2.0*M_PI*var[1])*exp(-fabs(diff[1]*diff[1])/(2.0*var[1]));
  const float h_score = exp(-fabs(diff[0]*diff[0])/(2.0*var[0]));
  const float s_score = exp(-fabs(diff[1]*diff[1])/(2.0*var[1]));
  // const float v_score = exp(-fabs(diff[2]*diff[2])/(2.0*var[2]));
  return (h_score+s_score)/2.0;
}

float ArmObjSegmentation::getUnaryWeight(cv::Vec3f sample, cv::Vec3f fg_mean, cv::Vec3f fg_var,
                                         cv::Vec3f bg_mean, cv::Vec3f bg_var)
{
  cv::Vec3f fg_diff = sample - fg_mean;
  cv::Vec3f bg_diff = sample - bg_mean;
  const float h_score = (exp(-fabs(fg_diff[0]*fg_diff[0])/(2.0*fg_var[0])) /
                         exp(-fabs(bg_diff[0]*bg_diff[0])/(2.0*bg_var[0])));
  const float s_score = (exp(-fabs(fg_diff[1]*fg_diff[1])/(2.0*fg_var[1])) /
                         exp(-fabs(bg_diff[1]*bg_diff[1])/(2.0*bg_var[1])));
  // const float v_score = exp(-fabs(diff[2]*diff[2])/(2.0*var[2]));
  return (h_score+s_score)/2.0;
}
}; // namespace tabletop_pushing
