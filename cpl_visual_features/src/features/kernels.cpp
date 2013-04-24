/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Georgia Institute of Technology
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

#include <cpl_visual_features/features/kernels.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace cpl_visual_features
{

double chiSqaureKernel(std::vector<double>& x, std::vector<double>& y)
{
  double kernel_sum = 0.0;
  for (unsigned int i = 0; i < x.size(); ++i)
  {
    double numerator = x[i]-y[i];
    if (x[i] == 0.0 && y[i] == 0.0)
    {
      continue;
    }
    kernel_sum += (numerator*numerator)/(x[i]+y[i]);
  }
  return kernel_sum;
}

cv::Mat chiSquareKernelBatch(cv::Mat& x, cv::Mat& y, double gamma)
{
  cv::Mat K(cv::Size(x.rows, y.rows), CV_64FC1, cv::Scalar(0.0));
  // Compute chi-square elementwise
  for (int i = 0; i < x.rows; ++i)
  {
    std::vector<double> xr = x.row(i);
    for (int j = 0; j < y.rows; ++j)
    {
      std::vector<double> yr = y.row(i);
      K.at<double>(i,j) = cpl_visual_features::chiSqaureKernel(xr, yr);
    }
  }
  if (gamma > 0.0)
  {
    cv::Mat exp_K;
    cv::exp(-gamma*K, exp_K);
    return exp_K;
  }
  return K;
}

};
