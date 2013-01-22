#include <cpl_visual_features/features/shape_context.h>
#include <cpl_visual_features/extern/lap_cpp/lap.h>
#include <math.h>
#include <string>
#include <iostream>

namespace cpl_visual_features
{
double compareShapes(cv::Mat& imageA, cv::Mat& imageB, double epsilonCost, bool write_images, std::string filePath, int max_displacement)
{
  cv::Mat edge_imageA(imageA.size(), imageA.type());
  cv::Mat edge_imageB(imageB.size(), imageB.type());

  // do edge detection
  cv::Canny(imageA, edge_imageA, 0.05, 0.5);
  cv::Canny(imageB, edge_imageB, 0.05, 0.5);
  if (write_images)
  {
    cv::imwrite((filePath+"/edge_imageA_raw.bmp").c_str(), edge_imageA);
    cv::imwrite((filePath+"/edge_imageB_raw.bmp").c_str(), edge_imageB);
  }
  // sample a subset of the edge pixels
  Samples samplesA = samplePoints(edge_imageA);
  Samples samplesB = samplePoints(edge_imageB);
  // construct shape descriptors for each sample
  ShapeDescriptors descriptorsA = constructDescriptors(samplesA);
  ShapeDescriptors descriptorsB = constructDescriptors(samplesB);
  cv::Mat cost_matrix = computeCostMatrix(descriptorsA, descriptorsB,
                                          epsilonCost, write_images, filePath);
  // save the result
  if (write_images)
  {
    cv::imwrite((filePath+"/edge_imageA.bmp").c_str(), edge_imageA);
    cv::imwrite((filePath+"/edge_imageB.bmp").c_str(), edge_imageB);
  }

  // do bipartite graph matching to find point correspondences
  // (uses code from http://www.magiclogic.com/assignment.html)
  Path min_path;
  double score = getMinimumCostPath(cost_matrix, min_path);
  displayMatch(edge_imageA, edge_imageB, samplesA, samplesB, min_path, max_displacement, filePath);
  int sizeA = samplesA.size();
  int sizeB = samplesB.size();

  // TODO: Return correspondences as well
  return (score-(fabs(sizeA-sizeB)*epsilonCost));
}

Samples samplePoints(cv::Mat& edge_image, double percentage)
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

ShapeDescriptors constructDescriptors(Samples& samples,
                                      unsigned int radius_bins,
                                      unsigned int theta_bins)
{
  ShapeDescriptors descriptors;
  ShapeDescriptor descriptor;
  double max_radius = 0;
  double radius, theta;
  double x1, x2, y1, y2;
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
                          ShapeDescriptors& descriptorsB,
                          double epsilonCost,
                          bool write_images,
                          std::string filePath)
{
  int mat_size = std::max(descriptorsA.size(), descriptorsB.size());
  cv::Mat cost_matrix(mat_size, mat_size, CV_64FC1, 0.0f);
  double d_cost, hi, hj;
  ShapeDescriptor& descriptorA = descriptorsA.front();
  ShapeDescriptor& descriptorB = descriptorsB.front();

  // initialize cost matrix for dummy values
  for (int i=0; i < cost_matrix.rows; i++)
  {
    for (int j=0; j < cost_matrix.cols; j++)
    {
      cost_matrix.at<double>(i,j) = epsilonCost;
    }
  }

  for (unsigned int i=0; i < descriptorsA.size(); i++)
  {
    descriptorA = descriptorsA.at(i);
    for (unsigned int j=0; j < descriptorsB.size(); j++)
    {
      descriptorB = descriptorsB.at(j);
      d_cost = 0;

      // compute cost between shape context i and j
      // using chi-squared test statistic
      for (unsigned int k=0; k < descriptorA.size(); k++)
      {
        hi = descriptorA.at(k) / (descriptorsA.size() - 1); // normalized bin val
        hj = descriptorB.at(k) / (descriptorsB.size() - 1); // normalized bin val
        if (hi + hj > 0)
        {
          d_cost += pow(hi-hj, 2) / (hi + hj);
        }
      }
      d_cost /= 2;
      cost_matrix.at<double>(i,j) = d_cost;
    }
  }

  cv::Mat int_cost_matrix;
  cost_matrix.convertTo(int_cost_matrix, CV_8UC1, 255);
  if (write_images)
  {
    cv::imwrite((filePath+"/cost_matrix.bmp").c_str(), int_cost_matrix);
  }

  return cost_matrix;
  // return int_cost_matrix;
}

double getMinimumCostPath(cv::Mat& cost_matrix, Path& path)
{
  const int dim = cost_matrix.rows;
  LapCost **cost_mat;
  cost_mat = new LapCost*[dim];
  // std::cout << "Allocating cost matrix" << std::endl;
  for (int r = 0; r < dim; ++r)
  {
    cost_mat[r] = new LapCost[dim];
  }
  // std::cout << "Populating cost matrix" << std::endl;
  for (int r = 0; r < dim; ++r)
  {
    for (int c = 0; c < dim; ++c)
    {
      cost_mat[r][c] = cost_matrix.at<double>(r,c);
    }
  }
  LapRow* rowsol;
  LapCol* colsol;
  LapCost* u;
  LapCost* v;
  rowsol = new LapCol[dim];
  colsol = new LapRow[dim];
  u = new LapCost[dim];
  v = new LapCost[dim];
  // std::cout << "Running lap" << std::endl;
  LapCost match_cost = lap(dim, cost_mat, rowsol, colsol, u, v);
  // std::cout << "Ran lap" << std::endl;
  for (int r = 0; r < dim; ++r)
  {
    int c = rowsol[r];
    path.push_back(c);
  }
  // std::cout << "Converted lap result" << std::endl;
  for (int r = 0; r < dim; ++r)
  {
    delete cost_mat[r];
  }
  delete cost_mat;
  delete u;
  delete v;
  delete rowsol;
  delete colsol;
  return match_cost;
}

void displayMatch(cv::Mat& edge_imageA, cv::Mat& edge_imageB,
                  Samples& samplesA, Samples& samplesB,
                  Path& path, int max_displacement, std::string filePath)
{
  cv::Mat disp_img;
  edge_imageA.copyTo(disp_img);
  // Display image B as another color
  cv::cvtColor(disp_img, disp_img, CV_GRAY2BGR);
  for (int r=0; r < disp_img.rows; r++)
  {
    for (int c=0; c < disp_img.cols; c++)
    {
      if (edge_imageB.at<uchar>(r,c) > 0)
        disp_img.at<cv::Vec3b>(r,c) = cv::Vec3b(0,0,255);
    }
  }

  for (unsigned int i = 0; i < samplesA.size(); ++i)
  {
    cv::Point start_point = samplesA[i];
    cv::Point end_point = samplesB[path[i]];
    if (std::abs(start_point.x - end_point.x) +
        std::abs(start_point.y - end_point.y) < max_displacement &&
        end_point.x > 0 && end_point.x < edge_imageB.rows &&
        end_point.y > 0 && end_point.x < edge_imageB.cols)
    {
      cv::line(disp_img, start_point, end_point, cv::Scalar(0,255,0));
    }
  }
  cv::imwrite((filePath+"/matches.bmp").c_str(), disp_img);
  cv::imshow("match", disp_img);
  cv::imshow("edgesA", edge_imageA);
  cv::imshow("edgesB", edge_imageB);
  cv::waitKey(3);
}
};
