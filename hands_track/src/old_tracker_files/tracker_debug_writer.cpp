#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <sstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cv.h>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;

//declare frame globally
Mat glob_img_of_interest;

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
bool get_centroid(const Mat img_of_interest, const int method, const float color_means[], const float color_var[], const Mat color_cov_mat_inv, Point2f*  out_blob_centroid, double area_thresh)
{
  //convert to binary image
  Mat bin_img(img_of_interest.rows, img_of_interest.cols, CV_32F);
  
  if(method == 0)
    {
      comparison(img_of_interest, color_means, color_var, bin_img);
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
  
  /*//debug - show image
  if(color_means[0]<100.0)
    {
      namedWindow("Binary Image", 1);
      imshow("Binary Image", bin_img);
      waitKey(0);   
      }*/
  if ((max_idx>-1) && (max_area>area_thresh))
    {
      Point2f cont_centr;
      float  cont_rad=0;
      minEnclosingCircle(contours[max_idx], cont_centr, cont_rad);
      //return the centroid computed
      *out_blob_centroid = cont_centr;
      return true;
    }

  else{return false;}
       
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

void kin_rgb_fram_sub(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
    try
      {
	cv_ptr = cv_bridge::toCvCopy(msg,  sensor_msgs::image_encodings::BGR8);
      }
    catch (cv_bridge::Exception& e)
      {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
      }
    
    //    namedWindow("show",1);

    glob_img_of_interest = cv_ptr->image;
    //cout<<"Who gets out first\n";
    //imshow("show", glob_img_of_interest);
    //waitKey(0);
}


//MAIN
int main(int argc, char **argv)
{
  //videoWrite
  VideoWriter outputVideo;
  bool first = true;

  //Initialize ROS
  ros::init(argc, argv, "hand_tracker");
  ros::NodeHandle hand_track_node;
  image_transport::ImageTransport hand_tracking(hand_track_node);
  
  int frame_count = 0;

  string input_file_green = "src/green.txt", input_file_blue="src/blue.txt";

  //read blue variables
  /*  float blue_means[2], blue_var[2], blue_hs_cov;
  read_file(input_file_blue, blue_means, blue_var, &blue_hs_cov);
  
  //debug
  cout<<blue_means[0]<<endl<<blue_means[1]<<endl<<blue_var[0]<<endl<<blue_var[1]<<endl<<blue_hs_cov<<endl;

  //read green variables
  float green_means[2], green_var[2], green_hs_cov;
  read_file(input_file_green, green_means, green_var, &green_hs_cov);
  
  //debug
  cout<<green_means[0]<<endl<<green_means[1]<<endl<<green_var[0]<<endl<<green_var[1]<<endl<<green_hs_cov<<endl;
  
  //covariance matrix and its inverse
  float blue_cov_mat[2][2] = {{blue_var[0], blue_hs_cov}, {blue_hs_cov, 
							   blue_var[1]}};
  Mat blue_cov_mat_inv = Mat(2,2,CV_32F,blue_cov_mat).inv(); 
  cout<< blue_cov_mat_inv;

  float green_cov_mat[2][2] = {{green_var[0], green_hs_cov}, {green_hs_cov, 
							      green_var[1]}};
  Mat green_cov_mat_inv = Mat(2,2,CV_32F,green_cov_mat).inv(); 
  cout<< green_cov_mat_inv;*/
   
  //Movie Window
  namedWindow( "Track Hands", 2 );
  int delay = 5;
  char c;
  
  //start looking at new images to find the colors tracked - loop here
  int frames_to_hold =10;
  Scalar cross_color1(0, 235, 0);
  Scalar cross_color2(0, 0, 245);
  
  image_transport::Subscriber kinect_rgb_sub = hand_tracking.subscribe
    ("kinect0/rgb/image_rect_color", frames_to_hold, &kin_rgb_fram_sub);
  
  //  if(glob_img_of_interest.empty()){cout<<"Then say its empty..\n";}
 
  while(ros::ok())
    {
      ros::spinOnce();
      Mat img_of_interest = glob_img_of_interest;
      
      if(img_of_interest.empty()){}//cout<< "\n Its empty"<<endl;}
      else
	{
	  //video
	  if(first){outputVideo.open("kinect_stuff.avi",CV_FOURCC('P','I','M','1'),30.0, glob_img_of_interest.size()) ;first=false;if(!outputVideo.isOpened()){cout<<"Cant Open Video";exit(-1);}}
	  
	  /*switch(frame_count)
	    {
	    case 200:  imwrite("imgs/1.jpg", img_of_interest);
	      break;
	    case 300:  imwrite("img/2.jpg", img_of_interest);
	      break; 
	    case 400:  imwrite("imgs/3.jpg", img_of_interest);
	      break;
	    case 500:  imwrite("imgs/4.jpg", img_of_interest);
	      break;
	    case 600:  imwrite("imgs/5.jpg", img_of_interest);
	      break;
	    case 700:  imwrite("imgs/6.jpg", img_of_interest);
	      break;
	    case 800:  imwrite("imgs/7.jpg", img_of_interest);
	      break;
	    case 900:  imwrite("imgs/8.jpg", img_of_interest);
	      break;
	    case 1000:  imwrite("imgs/9.jpg", img_of_interest);
	    exit(-1);}*/
	  outputVideo<<img_of_interest;
	  //debug
	  //cout<<endl<<"Do you every reach?"<<endl;
	  //Mat img_of_interest;
      
	  //Mat img_of_interest=  imread("src/data/blue_green3.jpg", CV_LOAD_IMAGE_COLOR);
  
	  //debug - start counting time
	  //double t = (double)getTickCount();
  
	  //parameters for comparison
	  int blue_method = 0, green_method=0;
	  double blue_threshold, green_threshold;
	  Point2f blue_centroid, green_centroid;
	  double blue_area_thresh = 25.0, green_area_thresh = 25.0;

	  /*//call function for centroid computation
	  bool found_blue = get_centroid(img_of_interest, blue_method, blue_means, 
					 blue_var, blue_cov_mat_inv, &blue_centroid, blue_area_thresh);
	  bool found_green = get_centroid(img_of_interest, green_method, green_means, 
	  green_var, green_cov_mat_inv, &green_centroid, green_area_thresh);*/
	  //debug - stop counting time
	  //t = ((double)getTickCount() - t)/getTickFrequency();
  

	  //debug- draw a cross to denote centroids

	  /*	  if(found_blue){draw_centroid(img_of_interest, blue_centroid, cross_color1);}
	  else{cout<<"\nCouldn't find blue hand\n";}
	  if(found_green){draw_centroid(img_of_interest, green_centroid, cross_color2);}
	  else{cout<<"\nCouldn't find green hand\n";}*/

	  //debug
	  //cout<<"I'm at least reaching\n";

	  imshow( "Track Hands", img_of_interest );
	  c = (char)cvWaitKey(delay);
	  if (c == 27) exit(-1);
	  
	  //debug - display execution time
	  //cout << "\nTimes passed in seconds: " << t << endl;
	  
	  //debug record a few frames
	  /*switch (frame_count)
	    {
	      imwrite("letsSeeGreen.jpg", img_of_interest);
	    }
	    cout<<"\nOOOOOOOOOOO LOOK!!\n"<<frame_count++;
	    }*/
    }
      return 0;
}
}
