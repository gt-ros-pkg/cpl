/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Georgia Institute of Technology
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Georgia Institute of Technology nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#ifndef cpl_visual_features_helpers_h_DEFINED
#define cpl_visual_features_helpers_h_DEFINED

#include <iostream>
#include <fstream>

#include <math.h>

#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace cpl_visual_features
{

typedef float (*kernel)(const float&, const float&);

cv::Mat downSample(cv::Mat data_in, int scales)
{
  cv::Mat out = data_in.clone();
  for (int i = 0; i < scales; ++i)
  {
    cv::pyrDown(data_in, out);
    data_in = out;
  }
  return out;
}

cv::Mat upSample(cv::Mat data_in, int scales)
{
  cv::Mat out = data_in.clone();
  for (int i = 0; i < scales; ++i)
  {
    // NOTE: Currently assumes even cols, rows for data_in
    cv::Size out_size(data_in.cols*2, data_in.rows*2);
    cv::pyrUp(data_in, out, out_size);
    data_in = out;
  }
  return out;
}

double subPIAngle(double theta)
{
  while (theta > M_PI)
  {
    theta -= 2.0*M_PI;
  }
  while (theta < -M_PI)
  {
    theta += 2.0*M_PI;
  }
  return theta;
}

float Cubic(const float& x, const float& scale)
{
  /*
  % See Keys, "Cubic Convolution Interpolation for Digital Image
  % Processing," IEEE Transactions on Acoustics, Speech, and Signal
  % Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.
  */

  float absx = fabs(x*scale);
  float absx2 = pow(absx, 2);
  float absx3 = pow(absx, 3);

  float f = (1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) *
            ((1 < absx) & (absx <= 2));

  return f*scale;
}


void Contributions(const unsigned int& in_length, const unsigned int& out_length, const float& scale,
      /*std::function<float (const float&)>*/kernel KernelFoo, const float& kernel_scale, float kernel_width,
      const bool antialiasing, boost::shared_array<float>& weights, boost::shared_array<int>& indices,
      unsigned int& P)
{
  float sum;
  float* weights_ptr, *weights_temp_ptr;
  int* indices_ptr, *indices_temp_ptr;

  P = ceil(kernel_width) + 2;

  boost::scoped_array<bool> kill_col(new bool[P]);
  for (unsigned int y = 0; y < P; y++) {
    kill_col[y] = true;
  }

  weights.reset(new float[out_length*P]);
  indices.reset(new int[out_length*P]);

  for (unsigned int x = 1; x <= out_length; x++) {
    float u = x/scale + (0.5 * (1 - 1/scale));
    float left = floor(u - kernel_width/2);

    sum = 0.0;

    weights_ptr = weights.get() + ((x-1)*P);
    indices_ptr = indices.get() + ((x-1)*P);

    for (unsigned int y = 0; y < P; y++) {
      indices_ptr[y] = left + y - 1;

      weights_ptr[y] = KernelFoo(u - (indices_ptr[y] + 1), kernel_scale);
      sum += weights_ptr[y];

      if (indices_ptr[y] < 0)
        indices_ptr[y] = 0;
      else if(indices_ptr[y] >= in_length)
        indices_ptr[y] = in_length - 1;
    }

    for (unsigned int y = 0; y < P; y++) {
      weights_ptr[y] /= sum;

      kill_col[y] = (kill_col[y] && (weights_ptr[y] == 0));
    }
  }


  // Remove the kill columns from both weights and indices
  boost::scoped_array<unsigned int> relocate_idx(new unsigned int[P]);
  unsigned int max_idx = 0;
  for (unsigned int y = 0; y < P; y++) {
    if (!kill_col[y])
      relocate_idx[max_idx++] = y;
  }

  if (max_idx < P) {
    boost::shared_array<float> weights_temp(new float[out_length*max_idx]);
    boost::shared_array<int> indices_temp(new int[out_length*max_idx]);

    for (unsigned int x = 0; x < out_length; x++) {
      weights_ptr = weights.get() + (x*P);
      indices_ptr = indices.get() + (x*P);

      weights_temp_ptr = weights_temp.get() + (x*max_idx);
      indices_temp_ptr = indices_temp.get() + (x*max_idx);

      for (unsigned int y = 0; y < max_idx; y++) {
        weights_temp_ptr[y] = weights_ptr[relocate_idx[y]];
        indices_temp_ptr[y] = indices_ptr[relocate_idx[y]];
      }
    }

    weights = weights_temp;
    indices = indices_temp;
  }

  P = max_idx;
}


template <class T>
void ResizeAlongDim(const cv::Mat& in, const unsigned int& dim, const boost::shared_array<float>& weights, const boost::shared_array<int>& indices, const unsigned int& out_length, const unsigned int& P, cv::Mat& out)
{
  float sum;
  int index;
  unsigned int in_gap_next_pxl, out_gap_next_pxl;
  unsigned int limit;

  if (dim == 0) {
    // for height rescaling

    out = cv::Mat::zeros(out_length, in.cols, in.type());
    limit = in.cols;

    in_gap_next_pxl = in.cols*in.channels();
    out_gap_next_pxl = out.cols*in.channels();
  } else {
    // for width rescaling

    out = cv::Mat::zeros(in.rows, out_length, in.type());
    limit = in.rows;

    in_gap_next_pxl = in.channels();
    out_gap_next_pxl = in.channels();
  }

  T* in_ptr = (T*)in.data;
  T* out_ptr = (T*)out.data;
  unsigned int ch = in.channels();

  for (unsigned int k = 0; k < limit; k++) {
    for (unsigned int m = 0; m < out_length; m++) {
      for (unsigned int c = 0; c < ch; c++) {
        sum = 0.0;

        for (unsigned int p = 0; p < P; p++) {
          index = indices[(m*P) + p];
          sum += weights[(m*P) + p] * in_ptr[(index * in_gap_next_pxl) + c];
        }

        out_ptr[(m * out_gap_next_pxl) + c] = sum; //(sum < 0 ? 0 : sum);
      }
    }

    if (dim == 0) {
      in_ptr += ch;
      out_ptr += ch;
    } else {
      in_ptr += in.cols*ch;
      out_ptr += out.cols*ch;
    }
  }
}

void imResize(const cv::Mat& in_im, const float& scale, cv::Mat& out_im)
{
  cv::Mat src_im, intermediate_out;
  float kernel_scale;
  float kernel_width = 4;
  bool antialiasing = true;
  unsigned int rscl_h = ceil(in_im.rows*scale);
  unsigned int rscl_w = ceil(in_im.cols*scale);
  unsigned int P;

  // convert input matrix to float
  int src_type = in_im.type();
  in_im.convertTo(src_im, CV_32FC(in_im.channels()));

  boost::shared_array<float> weights;
  boost::shared_array<int> indices;

  if ((scale < 1) && antialiasing) {
    // Use a modified kernel to simultaneously interpolate and
    // antialias.
    //auto fooH = [&](const float& x) -> float { return (scale * Cubic(scale*x)); };
    kernel_width = kernel_width / scale;
    kernel_scale = scale;
  } else {
    // No antialiasing; use unmodified kernel.
    kernel_scale = 1.0;
  }

  Contributions(src_im.rows, rscl_h, scale, &Cubic, kernel_scale, kernel_width, antialiasing, weights, indices, P);

  if (src_im.type() == CV_8UC1 || src_im.type() == CV_8UC3)
    ResizeAlongDim<unsigned char>(src_im, 0, weights, indices, rscl_h, P, intermediate_out);
  else
    ResizeAlongDim<float>(src_im, 0, weights, indices, rscl_h, P, intermediate_out);

  Contributions(src_im.cols, rscl_w, scale, &Cubic, kernel_scale, kernel_width, antialiasing, weights, indices, P);

  if (src_im.type() == CV_8UC1 || src_im.type() == CV_8UC3)
    ResizeAlongDim<unsigned char>(intermediate_out, 1, weights, indices, rscl_w, P, out_im);
  else
    ResizeAlongDim<float>(intermediate_out, 1, weights, indices, rscl_w, P, out_im);

  /*
  int x = 10, y = 1;
  std::cout << out_im.at<cv::Vec3f>(x,y)[0] << "," << out_im.at<cv::Vec3f>(x,y)[1] << "," << out_im.at<cv::Vec3f>(x,y)[2] << std::endl;

  // if the mat is going to be converted to unsigned char - prepare for rounding
  if (src_type == CV_8UC1) {
    out_im += cv::Scalar(1e-5);
  } else if (src_type == CV_8UC3) {
    out_im += cv::Scalar(1e-5, 1e-5, 1e-5);
  }

  std::cout << out_im.at<cv::Vec3f>(x,y)[0] << "," << out_im.at<cv::Vec3f>(x,y)[1] << "," << out_im.at<cv::Vec3f>(x,y)[2] << std::endl;
  */

  // convert output to original src type
  out_im.convertTo(out_im, src_type);

  //std::cout << (int)out_im.at<cv::Vec3b>(x,y)[0] << "," << (int)out_im.at<cv::Vec3b>(x,y)[1] << "," << (int)out_im.at<cv::Vec3b>(x,y)[2] << std::endl;
}

};
#endif // cpl_visual_features_helpers_h_DEFINED
