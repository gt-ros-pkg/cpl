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
#include <cpl_visual_features/comp_geometry.h>

namespace cpl_visual_features
{
bool lineSegmentIntersection2D(pcl16::PointXYZ a1, pcl16::PointXYZ a2, pcl16::PointXYZ b1, pcl16::PointXYZ b2,
                               pcl16::PointXYZ& intersection)
{
  if (!lineLineIntersection2D(a1, a2, b1, b2, intersection))
  {
    return false;
  }
  // Test if intersection is between a1 and a2
  return (pointIsBetweenOthers(intersection, a1, a2) && pointIsBetweenOthers(intersection, b1, b2));
}

bool lineLineIntersection2D(pcl16::PointXYZ a1, pcl16::PointXYZ a2, pcl16::PointXYZ b1, pcl16::PointXYZ b2,
                            pcl16::PointXYZ& intersection)
{
  float denom = (a1.x-a2.x)*(b1.y-b2.y)-(a1.y-a2.y)*(b1.x-b2.x);
  if (denom == 0) // Parrallel lines
  {
    return false;
  }
  intersection.x = ((a1.x*a2.y - a1.y*a2.x)*(b1.x-b2.x) -
                    (a1.x - a2.x)*(b1.x*b2.y - b1.y*b2.x))/denom;
  intersection.y = ((a1.x*a2.y - a1.y*a2.x)*(b1.y-b2.y) -
                    (a1.y - a2.y)*(b1.x*b2.y - b1.y*b2.x))/denom;
  return true;
}

bool pointIsBetweenOthers(pcl16::PointXYZ& pt, pcl16::PointXYZ& x1, pcl16::PointXYZ& x2)
{
  return (pt.x >= std::min(x1.x, x2.x) && pt.x <= std::max(x1.x, x2.x) &&
          pt.y >= std::min(x1.y, x2.y) && pt.y <= std::max(x1.y, x2.y));
}

};
