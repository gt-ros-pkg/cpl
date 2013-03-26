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
#ifndef cpl_comp_geometry_h_DEFINED
#define cpl_comp_geometry_h_DEFINED

#include <ros/ros.h>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <geometry_msgs/PointStamped.h>

namespace cpl_visual_features
{

bool lineSegmentIntersection2D(pcl16::PointXYZ a1, pcl16::PointXYZ a2, pcl16::PointXYZ b1, pcl16::PointXYZ b2,
                               pcl16::PointXYZ& intersection);

bool lineLineIntersection2D(pcl16::PointXYZ a1, pcl16::PointXYZ a2, pcl16::PointXYZ b1, pcl16::PointXYZ b2,
                            pcl16::PointXYZ& intersection);

bool pointIsBetweenOthers(pcl16::PointXYZ& pt, pcl16::PointXYZ& x1, pcl16::PointXYZ& x2);

static inline double dist(pcl16::PointXYZ a, pcl16::PointXYZ b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  const double dz = a.z-b.z;
  return std::sqrt(dx*dx+dy*dy+dz*dz);
}

static inline double dist(pcl16::PointXYZ a, geometry_msgs::Point b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  const double dz = a.z-b.z;
  return std::sqrt(dx*dx+dy*dy+dz*dz);
}

static inline double dist(geometry_msgs::Point b, pcl16::PointXYZ a)
{
  return dist(a,b);
}

static inline double sqrDist(Eigen::Vector3f& a, pcl16::PointXYZ& b)
{
  const double dx = a[0]-b.x;
  const double dy = a[1]-b.y;
  const double dz = a[2]-b.z;
  return dx*dx+dy*dy+dz*dz;
}


static inline double sqrDist(Eigen::Vector4f& a, Eigen::Vector4f& b)
{
  const double dx = a[0]-b[0];
  const double dy = a[1]-b[1];
  const double dz = a[2]-b[2];
  return dx*dx+dy*dy+dz*dz;
}

static inline double sqrDist(pcl16::PointXYZ a, pcl16::PointXYZ b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  const double dz = a.z-b.z;
  return dx*dx+dy*dy+dz*dz;
}

static inline double sqrDistXY(pcl16::PointXYZ a, pcl16::PointXYZ b)
{
  const double dx = a.x-b.x;
  const double dy = a.y-b.y;
  return dx*dx+dy*dy;
}

};
#endif // cpl_comp_geometry_h_DEFINED
