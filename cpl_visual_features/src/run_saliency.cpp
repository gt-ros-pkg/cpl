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

  for (int i = 0; i < count; i++)
  {
    std::stringstream filepath;
    std::stringstream depth_filepath;
    std::stringstream outpath;
    if (count == 1 && path != "")
    {
      filepath << path;
    }

    else if (path != "")
    {
      filepath << path << i << ".png";
      if (use_depth)
      {
        depth_filepath << path << i << "_depth.png";
        outpath << path << i << "_ic_depth.png";
      }
      else
      {
        outpath << path << i << "_ic.png";
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
    cv::imshow("frame", frame);
    Mat depth_frame;
    if (use_depth)
    {
      std::cout << "\tdepth loc: " << depth_filepath.str() << std::endl;
      depth_frame = cv::imread(depth_filepath.str(), CV_8UC1);
      cv::imshow("depth", depth_frame);
    }
    cv::waitKey();
    try
    {
      Mat saliency_map;
      if (use_depth)
      {
        saliency_map = csm(frame, depth_frame);
      }
      else
      {
        saliency_map = csm(frame, false);
      }
      cv::imshow("saliency", saliency_map);
      cv::waitKey();
      cv::imwrite(outpath.str(), saliency_map);
    }
    catch(cv::Exception e)
    {
      std::cerr << e.err << std::endl;
    }
  }

  return 0;
}
