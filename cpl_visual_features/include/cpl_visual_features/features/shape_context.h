// shape_context.cpp : Defines the entry point for the console application.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
void compareShapes(cv::Mat& imageA, cv::Mat& imageB);
void samplePoints(cv::Mat& edge_image, std::vector<cv::Point>& samples);
void constructDescriptors(std::vector<cv::Point>& samples,
                          std::vector< std::vector<float> >& descriptors);
cv::Mat computeCostMatrix(std::vector< std::vector<float> >& descriptorsA,
                          std::vector< std::vector<float> >& descriptorsB);
