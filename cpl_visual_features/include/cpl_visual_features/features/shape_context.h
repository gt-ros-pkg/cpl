// shape_context.cpp : Defines the entry point for the console application.
//

#ifndef shape_context_h_DEFINED
#define shape_context_h_DEFINED

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>

namespace cpl_visual_features
{
typedef std::vector<double> ShapeDescriptor;
typedef std::vector<ShapeDescriptor> ShapeDescriptors;
typedef std::vector<int> Path;
typedef std::vector<cv::Point> Samples;
typedef std::vector<cv::Point2f> Samples2f;

double compareShapes(cv::Mat& imageA, cv::Mat& imageB,
                     double epsilonCost = 9e5, bool write_images=false,
                     std::string filePath=".", int max_displacement=30,
                     std::string filePostFix="");

std::vector<cv::Point> samplePoints(cv::Mat& edge_image,
                                    double percentage = 0.3);

ShapeDescriptors extractDescriptors(cv::Mat& image);

ShapeDescriptors constructDescriptors(Samples2f& samples, cv::Point2f& center,
                                      bool use_center = false,
                                      int radius_bins = 5,
                                      int theta_bins = 12,
                                      double max_radius = 0,
                                      double scale = 100.0);
ShapeDescriptors constructDescriptors(Samples2f& samples,
                                      unsigned int radius_bins = 5,
                                      unsigned int theta_bins = 12);

cv::Mat computeCostMatrix(ShapeDescriptors& descriptorsA,
                          ShapeDescriptors& descriptorsB,
                          double epsilonCost = 9e5,
                          bool write_images=false,
                          std::string filePath=".", std::string filePostFix="");

double getMinimumCostPath(cv::Mat& cost_matrix, Path& path);

void displayMatch(cv::Mat& edge_imgA,
                  cv::Mat& edge_imgB,
                  Samples& samplesA,
                  Samples& samplesB,
                  Path& path, int max_displacement=30,
                  std::string filePath=".", std::string filePostFix="");

int getHistogramIndex(double radius, double theta, int radius_bins, int theta_bins);

template <class sample_type> ShapeDescriptors constructDescriptors(sample_type& samples,
                                                                   unsigned int radius_bins = 5,
                                                                   unsigned int theta_bins = 12)
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
    x1 = samples.at(i).x;
    y1 = samples.at(i).y;
    for (k=0; k < samples.size(); k++)
    {
      if (k != i)
      {
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
  // std::cout << "Got max_radius of: " << max_radius << std::endl;

  // build a descriptor for each sample
  for (i=0; i < samples.size(); i++)
  {
    // initialize descriptor
    descriptor.clear();
    for (j=0; j < radius_bins*theta_bins; j++)
    {
      descriptor.push_back(0);
    }
    x1 = samples.at(i).x;
    y1 = samples.at(i).y;

    // construct descriptor
    for (m=0; m < samples.size(); m++)
    {
      if (m != i)
      {
        // std::cout << "Constructing descriptor for (" << i << ", " << m << ")" << std::endl;
        x2 = samples.at(m).x;
        y2 = samples.at(m).y;

        radius = sqrt(pow(x1-x2,2) + pow(y1-y2,2));
        radius = log(radius);
        radius /= max_radius;
        theta = atan2(y1-y2,x1-x2);
        theta += M_PI;
        theta /= 2*M_PI;
        // std::cout << "Getting idx for (" << radius << ", " << theta << ")" << std::endl;
        int idx = getHistogramIndex(radius, theta, radius_bins, theta_bins);
        // std::cout << "Idx is: " << idx << std::endl;
        descriptor.at(idx)++;
      }
    }

    // add descriptor to std::vector of descriptors
    descriptors.push_back(descriptor);
  }

  return descriptors;
}

};
#endif // shape_context_h_DEFINED
