// shape_context.cpp : Defines the entry point for the console application.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
namespace cpl_visual_features
{
typedef std::vector<float> ShapeDescriptor;
typedef std::vector<ShapeDescriptor> ShapeDescriptors;
typedef std::vector<int> Path;
typedef std::vector<cv::Point> Samples;

double compareShapes(cv::Mat& imageA, cv::Mat& imageB,
                     float epsilonCost = 9e5, bool write_images=false);
std::vector<cv::Point> samplePoints(cv::Mat& edge_image,
                                    float percentage = 0.3);
ShapeDescriptors constructDescriptors(Samples& samples,
                                      unsigned int radius_bins = 5,
                                      unsigned int theta_bins = 12);
cv::Mat computeCostMatrix(ShapeDescriptors& descriptorsA,
                          ShapeDescriptors& descriptorsB,
                          float epsilonCost = 9e5,
                          bool write_images=false);
double getMinimumCostPath(cv::Mat& cost_matrix, Path& path);
void displayMatch(cv::Mat& edge_imgA,
                  std::vector<cv::Point>& samplesA,
                  Samples& samplesB,
                  Path& path, int max_displacement=30);
};
