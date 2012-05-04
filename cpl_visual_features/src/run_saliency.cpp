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

#include <opencv2/highgui/highgui.hpp>
#include <sstream>

#include <cpl_visual_features/saliency/center_surround.h>
#include <cpl_visual_features/features/gabor_filter_bank.h>
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND

using cv::Mat;
using cv::Rect;
using cv::Size;
using std::pair;
using std::vector;
using namespace cpl_visual_features;


int main(int argc, char** argv)
{
  srand(time(NULL));
  int count = 1;
  std::string path = "";
  bool use_depth = false;
  if (argc > 1)
    path = argv[1];

  if (argc > 2)
    count = atoi(argv[2]);

  if (argc > 3)
  {
    use_depth = true;
  }

  CenterSurroundMapper csm(2,3,3,4);

  // std::vector<int> is;
  // is.push_back(74);
  // is.push_back(70);
  // is.push_back(18);
  // is.push_back(10);
  // for (int k = 0; k < is.size(); k++)
  for (int i = 0; i < count; i++)
  {
    // int i = is[k];
    std::stringstream filepath;
    std::stringstream depth_filepath;
    std::stringstream outpath;
    if (count == 1 && path != "")
    {
      filepath << path;
    }

    else if (path != "")
    {
      filepath << path << "color" << std::max(i,0) << ".png";
      if (use_depth)
      {
        depth_filepath << path << std::max(i,0) << "_depth.png";
        outpath << path << std::max(i,0) << "_ic_depth.png";
      }
      else
      {
        outpath << path << "result_base/img_";
        if (i > 9)
        {
          outpath << "00";
        }
        else
        {
          outpath << "000";
        }
        outpath << std::max(i,0) << "_itti.png";
      }
    }
    else
    {
      //filepath << "/home/thermans/data/test_images/robot.jpg";
      filepath << "/home/thermans/data/test_images/saliency_test_frame.png";
      depth_filepath << "/home/thermans/data/test_images/saliency_test_depth_frame.png";
      use_depth = true;
    }

    std::cout << "Image " << i << std::endl;
    std::cout << "\tloc: " << filepath.str() << std::endl;
    Mat frame;
    frame = cv::imread(filepath.str());
    // Mat frame;
    cv::imshow("frame", frame);
    Mat depth_frame;
    if (use_depth)
    {
      std::cout << "\tdepth loc: " << depth_filepath.str() << std::endl;
      depth_frame = cv::imread(depth_filepath.str(), CV_8UC1);
      cv::imshow("depth", depth_frame);
    }
    cv::waitKey(3);
    try
    {
      Mat saliency_map;
      bool max_zero = true;
      if (use_depth)
      {
        saliency_map = csm(frame, depth_frame);
      }
      else
      {
        saliency_map = csm(frame, false);
      }
      // TODO: crop before writing
      cv::Rect crop_rect(10, 30, 591, 431);
      std::cout << "crop_rect.x " << crop_rect.x << std::endl;
      std::cout << "crop_rect.y " << crop_rect.y << std::endl;
      std::cout << "crop_rect.width " << crop_rect.width << std::endl;
      std::cout << "crop_rect.height " << crop_rect.height << std::endl;
      cv::Mat cropped_map = saliency_map(crop_rect);
      cv::imshow("saliency", saliency_map);
      cv::imshow("saliency cropped", cropped_map);
      double max_val = 0;
      double min_val= 0;
      cv::minMaxLoc(cropped_map, &min_val, &max_val);
      max_zero = (max_val == 0);
      cv::waitKey(3);
      cv::imwrite(outpath.str(), saliency_map);
    }
    catch(cv::Exception e)
    {
      std::cerr << e.err << std::endl;
    }
  }

  return 0;
}
