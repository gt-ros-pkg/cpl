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

void compareShapes(cv::Mat& imageA, cv::Mat& imageB);
std::vector<cv::Point> samplePoints(cv::Mat& edge_image);
ShapeDescriptors constructDescriptors(std::vector<cv::Point>& samples);
cv::Mat computeCostMatrix(std::vector< std::vector<float> >& descriptorsA,
                          std::vector< std::vector<float> >& descriptorsB);
};
