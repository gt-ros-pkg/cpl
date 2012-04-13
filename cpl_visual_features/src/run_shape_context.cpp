#include <cpl_visual_features/features/shape_context.h>

int main(int argc, char** argv)
{
  // load images from disk
  cv::Mat imageA;
  cv::Mat imageB;

  if (argc < 3)
  {
    imageA = cv::imread("/home/thermans/Desktop/video000.bmp");
    imageB = cv::imread("/home/thermans/Desktop/video001.bmp");
  }
  else
  {
    imageA = cv::imread(argv[1]);
    imageB = cv::imread(argv[2]);
  }
  cv::Mat imageA1(imageA.size(), imageA.type());
  cv::Mat imageB1(imageB.size(), imageB.type());

  // convert from color to grayscale, needed by Canny
  cv::cvtColor(imageA, imageA1, CV_RGB2GRAY);
  cv::cvtColor(imageB, imageB1, CV_RGB2GRAY);
  compareShapes(imageA1, imageB1);

  return 0;
}
