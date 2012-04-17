#include <cpl_visual_features/features/shape_context.h>
#include <cpl_visual_features/extern/lap_cpp/lap.h>
#include <math.h>
#include <ros/ros.h>

namespace cpl_visual_features
{
double compareShapes(cv::Mat& imageA, cv::Mat& imageB)
{
  cv::Mat edge_imageA(imageA.size(), imageA.type());
  cv::Mat edge_imageB(imageB.size(), imageB.type());

  // do edge detection
  cv::Canny(imageA, edge_imageA, 0.05, 0.5);
  cv::Canny(imageB, edge_imageB, 0.05, 0.5);

  // sample a subset of the edge pixels
  Samples samplesA = samplePoints(edge_imageA);
  Samples samplesB = samplePoints(edge_imageB);

  // construct shape descriptors for each sample
  ShapeDescriptors descriptorsA = constructDescriptors(samplesA);
  ShapeDescriptors descriptorsB = constructDescriptors(samplesB);

  cv::Mat cost_matrix = computeCostMatrix(descriptorsA, descriptorsB);

  // save the result
  cv::imwrite("/home/thermans/Desktop/edge_imageA.bmp", edge_imageA);
  cv::imwrite("/home/thermans/Desktop/edge_imageB.bmp", edge_imageB);

  // do bipartite graph matching to find point correspondences
  // (uses code from http://www.magiclogic.com/assignment.html)
  Path min_path;
  double score = getMinimumCostPath(cost_matrix, min_path);
  ROS_INFO_STREAM("Object match with score: " << score);
  displayMatch(edge_imageA, samplesA, samplesB, min_path);
  // TODO: Return correspondences as well
  return score;
}

Samples samplePoints(cv::Mat& edge_image)
{
  Samples samples;
  Samples all_points;
  cv::Scalar pixel;
  for (int y=0; y < edge_image.rows; y++)
  {
    for (int x=0; x < edge_image.cols; x++)
    {
      if (edge_image.at<uchar>(y, x) > 0)
      {
        all_points.push_back(cv::Point(x, y));
      }
    }
  }

  // set edge image to black
  edge_image = cv::Scalar(0);

  // subsample a percentage of all points
  float percentage = 0.1;
  int scale = 1 / percentage;
  for (unsigned int i=0; i < all_points.size(); i++)
  {
    if (i%scale == 0)
    {
      samples.push_back(all_points.at(i));
      edge_image.at<uchar>(samples.back().y, samples.back().x) = 255;
    }
  }
  return samples;
}

ShapeDescriptors constructDescriptors(Samples& samples)
{
  ShapeDescriptors descriptors;
  ShapeDescriptor descriptor;
  float max_radius = 0;
  float radius, theta;
  unsigned int radius_bins = 5;
  unsigned int theta_bins = 12;
  float x1, x2, y1, y2;
  unsigned int i, j, k, m;

  // find maximum radius for normalization purposes
  for (i=0; i < samples.size(); i++)
  {
    for (k=0; k < samples.size(); k++)
    {
      if (k != i)
      {
        x1 = samples.at(i).x;
        y1 = samples.at(i).y;
        x2 = samples.at(k).x;
        y2 = samples.at(k).y;

        radius = sqrt(pow(x1-x2,2) + pow(y1-y2,2));
        if (radius > max_radius)
        {
          max_radius = radius;
        }
      }
    }
  }
  max_radius = log(max_radius);

  // build a descriptor for each sample
  for (i=0; i < samples.size(); i++)
  {
    // initialize descriptor
    descriptor.clear();
    for (j=0; j < radius_bins*theta_bins; j++)
    {
      descriptor.push_back(0);
    }

    // construct descriptor
    for (m=0; m < samples.size()-1; m++)
    {
      if (m != i)
      {
        x1 = samples.at(i).x;
        y1 = samples.at(i).y;
        x2 = samples.at(m).x;
        y2 = samples.at(m).y;

        radius = sqrt(pow(x1-x2,2) + pow(y1-y2,2));
        radius = log(radius);
        radius /= max_radius;
        theta = atan(fabs(y1-y2) / fabs(x1-x2));
        theta += M_PI/2;
        if (y1-y2 < 0)
        {
          theta += M_PI;
        }
        theta /= 2*M_PI;
        descriptor.at((int)(radius*radius_bins) * (int)(theta*theta_bins))++;
      }
    }

    // add descriptor to std::vector of descriptors
    descriptors.push_back(descriptor);
  }

  return descriptors;
}

cv::Mat computeCostMatrix(ShapeDescriptors& descriptorsA,
                          ShapeDescriptors& descriptorsB)
{
  int mat_size = std::max(descriptorsA.size(), descriptorsB.size());
  cv::Mat cost_matrix(mat_size, mat_size, CV_32FC1, 0.0f);
  float epsilonCost = 9e5;
  float cost, hi, hj;
  ShapeDescriptor& descriptorA = descriptorsA.front();
  ShapeDescriptor& descriptorB = descriptorsB.front();

  // initialize cost matrix for dummy values
  for (int i=0; i < cost_matrix.rows; i++)
  {
    for (int j=0; j < cost_matrix.cols; j++)
    {
      cost_matrix.at<float>(i,j) = epsilonCost;
    }
  }

  for (unsigned int i=0; i < descriptorsA.size(); i++)
  {
    descriptorA = descriptorsA.at(i);
    for (unsigned int j=0; j < descriptorsB.size(); j++)
    {
      descriptorB = descriptorsB.at(j);
      cost = 0;

      // compute cost between shape context i and j
      // using chi-squared test statistic
      for (unsigned int k=0; k < descriptorA.size(); k++)
      {
        hi = descriptorA.at(k) / (descriptorsA.size() - 1); // normalized bin val
        hj = descriptorB.at(k) / (descriptorsB.size() - 1); // normalized bin val
        if (hi + hj > 0)
        {
          cost += pow(hi-hj, 2) / (hi + hj);
        }
      }
      cost /= 2;
      cost_matrix.at<float>(i,j) = cost;
    }
  }

  cv::Mat int_cost_matrix;
  cost_matrix.convertTo(int_cost_matrix, CV_8UC1, 255);
  cv::imwrite("/home/thermans/Desktop/cost_matrix.bmp", int_cost_matrix);

  //return cost_matrix;
  return int_cost_matrix;
}

double getMinimumCostPath(cv::Mat& cost_matrix, Path& path)
{
  const int dim = cost_matrix.rows;
  cost **cost_mat;
  cost_mat = new cost*[dim];
  for (int r = 0; r < dim; ++r)
  {
    cost_mat[r] = new cost[dim];
  }
  for (int r = 0; r < dim; ++r)
  {
    for (int c = 0; c < dim; ++c)
    {
      cost_mat[r][c] = cost_matrix.at<uchar>(r,c);
    }
  }
  row* rowsol;
  col* colsol;
  cost* u;
  cost* v;
  rowsol = new col[dim];
  colsol = new row[dim];
  u = new cost[dim];
  v = new cost[dim];

  cost lap_cost = lap(dim, cost_mat, rowsol, colsol, u, v);
  // checklap(dim, cost_mat, rowsol, colsol, u, v);
  for (int r = 0; r < dim; ++r)
  {
    int c = rowsol[r];
    path.push_back(c);
  }
  return lap_cost;
}

void displayMatch(cv::Mat& edge_imageA, Samples& samplesA, Samples& samplesB,
                  Path& path)
{
  cv::Mat disp_img;
  edge_imageA.copyTo(disp_img);
  for (unsigned int i = 0; i < samplesA.size(); ++i)
  {
    cv::Point start_point = samplesA[i];
    cv::Point end_point = samplesB[path[i]];
    if (std::abs(start_point.x - end_point.x) +
        std::abs(start_point.y - end_point.y) < 30)
    {
      cv::line(disp_img, start_point, end_point, cv::Scalar(255,255,255));
    }
  }
  cv::imshow("match", disp_img);
  cv::imshow("edges", edge_imageA);
  cv::waitKey();
}
};
