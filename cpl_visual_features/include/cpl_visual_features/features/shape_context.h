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

double compareShapes(cv::Mat& imageA, cv::Mat& imageB,
                     double epsilonCost = 9e5, bool write_images=false,
                     std::string filePath=".", int max_displacement=30,
                     std::string filePostFix="");

std::vector<cv::Point> samplePoints(cv::Mat& edge_image,
                                    double percentage = 0.3);

ShapeDescriptors extractDescriptors(cv::Mat& image);

ShapeDescriptors constructDescriptors(Samples& samples,
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
};
#endif // shape_context_h_DEFINED
