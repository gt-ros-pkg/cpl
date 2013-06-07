#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cv.h>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;

//read file with means, variance and covariance
void read_file(const string file_name, float means[], float variance [], float* covariance)
{
  ifstream read_it(file_name.c_str());
  string dummy;
  stringstream holder;

  if(read_it.is_open())
    {
      read_it >> dummy;
      read_it >> dummy;
      read_it >> means[0];
      read_it >> dummy;
      read_it >> means[1];
      read_it >> dummy;
      read_it >> dummy;
      read_it >> variance[0];
      read_it >> dummy;
      read_it >> variance[1];
      read_it >> dummy;
      read_it >> dummy;
      read_it >> *covariance;

    }
   else{
	cout<< "Can't open file!";
	}
  
}

//Compares an image to a color in HS space- defined by mean and variance and returns a
//binary image with a morphological open performed
// image of interest expects(0-255)
// hsv_means = [h, s, v]
// bin_img = single channel CV_32F same size as image_of_interest
void comparison(const Mat img_of_interest, const float *hsv_means, const float* hs_variance, Mat bin_img)
{
  int close_iterations = 1;
  Mat ioi_hsv;
  cvtColor(img_of_interest, ioi_hsv, CV_BGR2HSV);
  //  cout<<"\nsize of hsv image :" <<ioi_hsv.rows<<','<<ioi_hsv.cols<<"  Type:"<<((ioi_hsv.type()==CV_8UC3)?"Yeah":"Nah")<<endl;
  
  //split into different matrices the channels
  Mat temp_h, temp_s, temp_v;
  vector<Mat> arr_mats;
  split(ioi_hsv, arr_mats);
  
  temp_h = arr_mats[0];
  temp_s = arr_mats[1];

  Mat diff_h(temp_h.size(), CV_32F);
  Mat tempor =  Mat::ones(temp_h.rows, temp_h.cols, CV_32F) * hsv_means[0];
  subtract(temp_h, tempor, diff_h, noArray(), CV_32F);
  diff_h = abs(diff_h);
  
  Mat diff_s(temp_h.size(), CV_32F);
  tempor =  Mat::ones(temp_h.rows, temp_h.cols, CV_32F) * hsv_means[1];
  subtract(temp_s, tempor, diff_s, noArray(), CV_32F);
  diff_s = abs(diff_s);
  
  
  //threshold
  //Mat bin_img(temp_h.rows, temp_h.cols, CV_32F);
  threshold(diff_h, diff_h, sqrt(hs_variance[0]), 1, THRESH_BINARY_INV);
  threshold(diff_s, diff_s, sqrt(hs_variance[1]), 1, THRESH_BINARY_INV);
  multiply(diff_h, diff_s, bin_img);
  
  //perform opening(erode, dilate)
  int struct_size = 3;
  Mat struc_elem = getStructuringElement(MORPH_ELLIPSE, Size(2*struct_size+1, 2*struct_size+1), Point (struct_size, struct_size));
  morphologyEx(bin_img, bin_img, MORPH_CLOSE, struc_elem, Point(-1,1), close_iterations);
}



//Compares an image to a color in HS space- computes mahabolonis distance for a diagonal matrix
//except the distance is weigted. So, the weight defines how much of the distance in S contributes to total distance
//binary image with a morphological open performed
// image of interest expects(0-255)
// hsv_means = [h, s, v]
// bin_img = single channel CV_32F same size as image_of_interest
void comparison_mahalanobis_diag(const Mat img_of_interest, const float *hsv_means, const float* hs_variance, Mat bin_img)
{
  //hard threshold
  double hard_thresh = 0.2;

  //saturation contribution
  double sat_contri = 0.0;
  
  //if saturation below threshold use only hue to generate binary
  float saturation_threshold = 0.0;
  
  int close_iterations = 1;
  Mat ioi_hsv, maha_dist;
  cvtColor(img_of_interest, ioi_hsv, CV_BGR2HSV);
  //  cout<<"\nsize of hsv image :" <<ioi_hsv.rows<<','<<ioi_hsv.cols<<"  Type:"<<((ioi_hsv.type()==CV_8UC3)?"Yeah":"Nah")<<endl;
  
  //split into different matrices the channels
  Mat temp_h, temp_s, temp_v;
  vector<Mat> arr_mats;
  split(ioi_hsv, arr_mats);
  
  temp_h = arr_mats[0];
  temp_s = arr_mats[1];
  temp_v = arr_mats[2];

  
  //double sigma, mean, s_thresh, v_thresh;
  //ros::param::param<double>("~sigma", sigma, 0.0026);
  // ros::param::param<double>("~mean", mean, 0.4731);
  //ros::param::param<double>("~s_thresh", s_thresh, 60);
  // ros::param::param<double>("~v_thresh", v_thresh, 10);
  //  threshold(-temp_s, temp_s, saturation_threshold, 

  double sigma =  0.0020;
  double mean =  73.22/180.0;//0.4731; 
  double  s_thresh= 50;
  double  v_thresh= 30; 

  Mat diff_h(temp_h.size(), CV_32F);
  Mat tempor =  Mat::ones(temp_h.rows, temp_h.cols, CV_32F) * mean * 180.0;
  subtract(temp_h, tempor, diff_h, noArray(), CV_32F);
  maha_dist = abs(diff_h);
  cout << "mean " << mean << endl;
  cout << "sigma " << sigma << endl;
  cout << "s_thresh " << s_thresh << endl;
  maha_dist = (1/sigma/180.0)*maha_dist;
  exp(-maha_dist, maha_dist);
  subtract(maha_dist, maha_dist, maha_dist, temp_s < s_thresh);
  subtract(maha_dist, maha_dist, maha_dist, temp_v < v_thresh);
  double max_value, min_value;
  minMaxLoc(maha_dist, &min_value, &max_value, NULL, NULL);
  cout<<"\nmax value:"<< max_value << "\nminvalue:"<<min_value<<endl;
  //normalize(maha_dist, maha_dist, 0.0,1.0, NORM_MINMAX);
  
  
  

  //threshold
  //Mat bin_img(temp_h.rows, temp_h.cols, CV_32F);
  threshold(maha_dist, bin_img, hard_thresh, 1.0, THRESH_BINARY);
  //threshold(diff_s, diff_s, sqrt(hs_variance[1]), 1, THRESH_BINARY_INV);
  //multiply(diff_h, diff_s, bin_img);


  //debug
  /*namedWindow("binary_image");
  imshow("binary_image", bin_img);
  waitKey(0);*/
  
  //perform opening(erode, dilate)
  int struct_size = 4;
  Mat struc_elem = getStructuringElement(MORPH_ELLIPSE, Size(2*struct_size+1, 2*struct_size+1), Point (struct_size, struct_size));
  morphologyEx(bin_img, bin_img, MORPH_CLOSE, struc_elem, Point(-1,1), close_iterations);

//debug
  namedWindow("binary_image");
  imshow("binary_image", bin_img);
  waitKey(0);
}


//fresh
void new_comparison_mahalanobis(const Mat img_of_interest, const float *hsv_means, const float* hs_variance, Mat bin_img, bool isGreen)
{

  double hard_thresh, sigma, mean, s_thresh, v_thresh;
  //hard threshold
  //double hard_thresh = 6.0;

  //saturation contribution
  double sat_contri = 0.0;
  
  //if saturation below threshold use only hue to generate binary
  float saturation_threshold = 0.0;
  
  int close_iterations = 1;
  Mat ioi_hsv, maha_dist;
  cvtColor(img_of_interest, ioi_hsv, CV_BGR2HSV);
  //  cout<<"\nsize of hsv image :" <<ioi_hsv.rows<<','<<ioi_hsv.cols<<"  Type:"<<((ioi_hsv.type()==CV_8UC3)?"Yeah":"Nah")<<endl;
  
  //split into different matrices the channels
  Mat temp_h, temp_s, temp_v;
  vector<Mat> arr_mats;
  split(ioi_hsv, arr_mats);
  
  temp_h = arr_mats[0];
  temp_s = arr_mats[1];
  temp_v = arr_mats[2];

  
  //double sigma, mean, s_thresh, v_thresh;
  //ros::param::param<double>("~sigma", sigma, 0.0026);
  // ros::param::param<double>("~mean", mean, 0.4731);
  //ros::param::param<double>("~s_thresh", s_thresh, 60);
  // ros::param::param<double>("~v_thresh", v_thresh, 10);
  //  threshold(-temp_s, temp_s, saturation_threshold, 

  /*//works for green
  double hard_thresh = 6.0;
  double sigma =  0.0026;
  double mean =  0.4731; 
  double  s_thresh= 100;
  double  v_thresh= 10; */

  /*  double hard_thresh = 6.0;
  double sigma =  0.0026;
  double mean =  0.4731; 
  double  s_thresh= 80;
  double  v_thresh= 30;*/
  
  if (isGreen)
    {
      hard_thresh = 6.0;
      sigma =  0.0026;
      mean =  80.33/180;
      s_thresh = 60;
      v_thresh= 50;
    }
  else
    {
      hard_thresh = 2.0;
      mean =  126.993/180;//0.4731; 
      s_thresh = 90;
      v_thresh= 50;
    }


  Mat diff_h(temp_h.size(), CV_32F);
  Mat tempor =  Mat::ones(temp_h.rows, temp_h.cols, CV_32F) * mean * 180.0;
  subtract(temp_h, temp_h, temp_h, temp_s < s_thresh);
  subtract(temp_h, temp_h, temp_h, temp_v < v_thresh);
  subtract(temp_h, tempor, diff_h, noArray(), CV_32F);
  maha_dist = abs(diff_h);
  
  Mat new_thresh = maha_dist + 2*hard_thresh*Mat::ones(maha_dist.rows, maha_dist.cols, CV_32F);

  subtract(new_thresh, maha_dist, maha_dist, temp_s<s_thresh);
  subtract(new_thresh, maha_dist, maha_dist, temp_v<v_thresh);
  cout << "mean " << mean << endl;
  cout << "sigma " << sigma << endl;
  cout << "s_thresh " << s_thresh << endl;
  //maha_dist = (1/sigma/180.0)*maha_dist;
  //exp(-maha_dist, maha_dist);
  //  subtract(maha_dist, maha_dist, maha_dist, temp_s > s_thresh);
  //subtract(maha_dist, maha_dist, maha_dist, temp_v > v_thresh);
  double max_value, min_value;
  minMaxLoc(maha_dist, &min_value, &max_value, NULL, NULL);
  cout<<"\nmax value:"<< max_value << "\nminvalue:"<<min_value<<endl;
  //normalize(maha_dist, maha_dist, 0.0,1.0, NORM_MINMAX);
  
  
  
  threshold(maha_dist, bin_img, hard_thresh, 1.0, THRESH_BINARY_INV);
  /* //debug
  namedWindow("binary_image");
  imshow("binary_image", bin_img);
  waitKey(0);*/

  //threshold
  //Mat bin_img(temp_h.rows, temp_h.cols, CV_32F);
  //threshold(maha_dist, bin_img, hard_thresh, 1.0, THRESH_BINARY);
  //threshold(diff_s, diff_s, sqrt(hs_variance[1]), 1, THRESH_BINARY_INV);
  //multiply(diff_h, diff_s, bin_img);


  //debug
  /*namedWindow("binary_image");
  imshow("binary_image", bin_img);
  waitKey(0);*/
  
  //perform opening(erode, dilate)
  int struct_size = 3;
  Mat struc_elem = getStructuringElement(MORPH_ELLIPSE, Size(2*struct_size+1, 2*struct_size+1), Point (struct_size, struct_size));
  morphologyEx(bin_img, bin_img, MORPH_CLOSE, struc_elem, Point(-1,1), close_iterations);

  namedWindow("binary_image");
  imshow("binary_image", bin_img);
  waitKey(0);

}




//Compares an image to a color in HS space- defined by mean and co-variance 
//matrix and returns a binary image with a morphological open performed
// image of interest expects(0-255)
// hsv_means = [h, s, v]
// bin_img = single channel CV_32F same size as image_of_interest
void mahalanobis_comparison(const Mat img_of_interest, const float *hsv_means, const Mat cov_mat_inv, Mat bin_img)
{
  int close_iterations = 2;
  Mat ioi_hsv;
  cvtColor(img_of_interest, ioi_hsv, CV_BGR2HSV);
  //  cout<<"\nsize of hsv image :" <<ioi_hsv.rows<<','<<ioi_hsv.cols<<"  Type:"<<((ioi_hsv.type()==CV_8UC3)?"Yeah":"Nah")<<endl;
  
  //split into different matrices the channels
  Mat temp_h, temp_s, temp_v;
  vector<Mat> arr_mats;
  split(ioi_hsv, arr_mats);
  temp_h = arr_mats[0];
  temp_s = arr_mats[1];
  
  int zero_pix = 0;
  
  //calculate mahalonobis dist for every pixel
  for (int i=0; i<temp_h.rows; i++)
    {
      const uchar* curr_ro_h  = temp_h.ptr<uchar>(i);
      const uchar* curr_ro_s  = temp_s.ptr<uchar>(i);
      //      double* curr_bin = bin_img.ptr<double>(i);
      
      for (int j=0; j<temp_h.cols; j++)
	{
	  bin_img.at<float>(i,j) = sqrt(abs(pow(float(curr_ro_h[j])-hsv_means[0],2)*cov_mat_inv.at<double>(0,0)+ 
					pow(float(curr_ro_s[j])-hsv_means[1],2)*cov_mat_inv.at<double>(1,1)+
					(float(curr_ro_h[j])-hsv_means[0])*(float(curr_ro_s[j])-hsv_means[1])*cov_mat_inv.at<double>(0,1) + (float(curr_ro_h[j])-hsv_means[0])*(float(curr_ro_s[j])-hsv_means[1])*cov_mat_inv.at<double>(1,0)));
	  
	 
	}
    }

  cout<<"\n Zeros = "<<zero_pix<<endl;
  //threshold
  threshold(bin_img, bin_img, 5.0e-12, 1, THRESH_BINARY_INV);
  
  //perform closing
  int struct_size = 3;
  Mat struc_elem = getStructuringElement(MORPH_ELLIPSE, Size(2*struct_size+1, 2*struct_size+1), Point (struct_size, struct_size));
  morphologyEx(bin_img, bin_img, MORPH_CLOSE, struc_elem, Point(-1,1), close_iterations);
}

//Finds color blob centroids using regular difference method(0) or 
//mahalanobis method(1) 
//false if no blob found
bool get_centroid(const Mat img_of_interest, const int method, const float color_means[], const float color_var[], const Mat color_cov_mat_inv, Point2f*  out_blob_centroid, double area_thresh, bool isGreen)
{
  //convert to binary image
  Mat bin_img(img_of_interest.rows, img_of_interest.cols, CV_32F);
  
  if(method == 0)
    {
      new_comparison_mahalanobis(img_of_interest, color_means, color_var, bin_img, isGreen);
      //      comparison_mahalanobis_diag(img_of_interest, color_means, color_var, bin_img);
    }
  else
    {
      mahalanobis_comparison (img_of_interest, color_means, color_cov_mat_inv, bin_img);
    }

  /*double max_value, min_value;
  minMaxLoc(bin_img, &min_value, &max_value, NULL, NULL);
  cout<<"\nmax value:"<< max_value << "\nminvalue:"<<min_value<<endl;*/
  
  //compute area and centroids
  vector<vector<Point> > contours;
  //  vector<Vec4i> hierarchy;
  Mat new_bin(bin_img.rows, bin_img.cols, CV_8U);
  bin_img.convertTo(new_bin, CV_8U, 255);
  
  findContours( new_bin, contours, noArray(),
        CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
  
  //iterate through all contours and find one with biggest area
  int max_idx = -1;
  double max_area = 0.0;
  double temp_cont_area = 0.0;
  for( int idx = 0; idx < contours.size(); idx++ )
    {
      temp_cont_area = contourArea(contours[idx]);
      if (temp_cont_area > max_area)
	{
	  max_area = temp_cont_area;
	  max_idx = idx;
	}
    }
  
  //debug - show image
  /*namedWindow("Binary Image", 1);
  imshow("Binary Image", bin_img);
  waitKey(0);*/ 

  if ((max_idx>-1) && (max_area>area_thresh))
    {
      Point2f cont_centr;
      float  cont_rad=0;
      minEnclosingCircle(contours[max_idx], cont_centr, cont_rad);
      //return the centroid computed
      *out_blob_centroid = cont_centr;
      return true;
    }

  else{cout<<"\nNot found blob\n";return false;}
       
}


void draw_centroid(Mat img_of_interest, Point2f in_centroid, const Scalar cross_color)
{

  int line_thickness = 5;
  Point centroid_1, centroid_2;
  centroid_1 = centroid_2 = in_centroid;
  centroid_1.x = std::max(0.f, in_centroid.x-10);
  centroid_2.x = std::min(float(img_of_interest.cols), in_centroid.x+10);
  line(img_of_interest, centroid_1, centroid_2, cross_color, line_thickness);

  centroid_1 = centroid_2 = in_centroid;
  centroid_1.y = std::max(0.f, in_centroid.y-10);
  centroid_2.y = std::min(float(img_of_interest.cols), in_centroid.y+10);
  line(img_of_interest, centroid_1, centroid_2, cross_color, line_thickness);

}


//MAIN
int main(int argc, char **argv)
{

  int frame_count = 0;

  string input_file_green = "green.txt", input_file_blue="blue.txt";

  //read blue variables
  float blue_means[2], blue_var[2], blue_hs_cov;
  read_file(input_file_blue, blue_means, blue_var, &blue_hs_cov);
  
  //debug
  //cout<<blue_means[0]<<endl<<blue_means[1]<<endl<<blue_var[0]<<endl<<blue_var[1]<<endl<<blue_hs_cov<<endl;

  //read green variables
  float green_means[2], green_var[2], green_hs_cov;
  read_file(input_file_green, green_means, green_var, &green_hs_cov);
  
  //debug
  //cout<<green_means[0]<<endl<<green_means[1]<<endl<<green_var[0]<<endl<<green_var[1]<<endl<<green_hs_cov<<endl;
  
  //covariance matrix and its inverse
  float blue_cov_mat[2][2] = {{blue_var[0], blue_hs_cov}, {blue_hs_cov, 
							   blue_var[1]}};
  Mat blue_cov_mat_inv = Mat(2,2,CV_32F,blue_cov_mat).inv(); 
  cout<< blue_cov_mat_inv;

  float green_cov_mat[2][2] = {{green_var[0], green_hs_cov}, {green_hs_cov, 
							      green_var[1]}};
  Mat green_cov_mat_inv = Mat(2,2,CV_32F,green_cov_mat).inv(); 
  cout<< green_cov_mat_inv;
  
  //Movie Window
  namedWindow( "Track Hands", 2 );
  int delay = 5;
  char c;
  
  //start looking at new images to find the colors tracked - loop here
  //  VideoCapture vid_of_interest("../data/gb_vid1.mov");
  VideoCapture vid_of_interest("kinect_stuff.avi");
  Scalar cross_color1(0, 235, 0);
  Scalar cross_color2(0, 0, 245);

  if (!vid_of_interest.isOpened())
    {
      cout<< "\nVideo Open Unsuccessful!"<<endl;
      exit(-1);
    }
  else
    {
      Mat img_of_interest;
      vid_of_interest >> img_of_interest;
      frame_count++;
      
      while(!img_of_interest.empty())
	{
	  while(frame_count<500)
	    {
	    vid_of_interest>>img_of_interest;
	    frame_count++;
	    }
	  //Mat img_of_interest;
      
	  //Mat img_of_interest=  imread("src/data/blue_green3.jpg", CV_LOAD_IMAGE_COLOR);
  
	  //debug - start counting time
	  //double t = (double)getTickCount();
  
	  //parameters for comparison
	  int blue_method = 0, green_method=0;
	  double blue_threshold, green_threshold;
	  Point2f blue_centroid, green_centroid;
	  double blue_area_thresh = 25.0, green_area_thresh = 25.0;

	  //call function for centroid computation
	  bool found_blue = get_centroid(img_of_interest, blue_method, blue_means, 
					 blue_var, blue_cov_mat_inv, &blue_centroid, blue_area_thresh, false);
	  bool found_green = get_centroid(img_of_interest, green_method, green_means, 
					  green_var, green_cov_mat_inv, &green_centroid, green_area_thresh, true)
	    ;
	  //debug - stop counting time
	  //t = ((double)getTickCount() - t)/getTickFrequency();
  

	  //debug- draw a cross to denote centroids

	  if(found_blue){draw_centroid(img_of_interest, blue_centroid, cross_color1);}
	  else{cout<<"\nCouldn't find blue hand\n";}
	  if(found_green){draw_centroid(img_of_interest, green_centroid, cross_color2);}
	  else{cout<<"\nCouldn't find green hand\n";}

	  imshow( "Track Hands", img_of_interest );
	  c = (char)cvWaitKey(delay);
	  if (c == 27) break;
	  
	  //debug - display execution time
	  //cout << "\nTimes passed in seconds: " << t << endl;
	  
	  //advance to next frame
	  vid_of_interest >> img_of_interest;
	  frame_count++;
	      
	  
	}
    }    

  return 0;
}
