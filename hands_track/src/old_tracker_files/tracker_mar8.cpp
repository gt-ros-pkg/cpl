#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_broadcaster.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/median.hpp>

#include <sstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cv.h>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace boost::accumulators;

typedef pcl::PointXYZRGB PRGB;

//declare frame globally
Mat glob_img_of_interest;
ros::Time last_img_time;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

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
  //if saturation below threshold use only hue to generate binary
  //float saturation_threshold = 0.0;
  
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

  
  //  threshold(-temp_s, temp_s, saturation_threshold,  

  Mat diff_h(temp_h.size(), CV_32F);
  Mat tempor =  Mat::ones(temp_h.rows, temp_h.cols, CV_32F) * hsv_means[0];
  subtract(temp_h, tempor, diff_h, noArray(), CV_32F);
  diff_h = abs(diff_h);
  
  Mat diff_s(temp_h.size(), CV_32F);
  tempor =  Mat::ones(temp_h.rows, temp_h.cols, CV_32F) * hsv_means[1];
  subtract(temp_s, tempor, diff_s, noArray(), CV_32F);
  diff_s = abs(diff_s);
  //cout<<"\n  the threshold::"<< (saturation_threshold - hsv_means[1]);
  //threshold saturation
  //threshold(diff_s, diff_s, (saturation_threshold - hsv_means[1]), 255.0, THRESH_TOZERO);
  //diff_s = abs(diff_s);

  //threshold
  //Mat bin_img(temp_h.rows, temp_h.cols, CV_32F);
  threshold(diff_h, diff_h, sqrt(hs_variance[0]), 1, THRESH_BINARY_INV);
  threshold(diff_s, diff_s, sqrt(hs_variance[1]), 1, THRESH_BINARY_INV);
  multiply(diff_h, diff_s, bin_img);
  
  //perform opening(erode, dilate)
  int struct_size = 2;
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
  double hard_thresh = 0.04;

  //saturation contribution
  double sat_contri = 0.0;
  
  //if saturation below threshold use only hue to generate binary
  float saturation_threshold = 0.0;
  
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

  
  //  threshold(-temp_s, temp_s, saturation_threshold,  

  Mat diff_h(temp_h.size(), CV_32F);
  Mat tempor =  Mat::ones(temp_h.rows, temp_h.cols, CV_32F) * hsv_means[0];
  subtract(temp_h, tempor, diff_h, noArray(), CV_32F);
  pow(diff_h, 2.0, diff_h);
  
  Mat diff_s(temp_h.size(), CV_32F);
  tempor =  Mat::ones(temp_h.rows, temp_h.cols, CV_32F) * hsv_means[1];
  subtract(temp_s, tempor, diff_s, noArray(), CV_32F);
  //  diff_s = abs(diff_s);
  //cout<<"\n  the threshold::"<< (saturation_threshold - hsv_means[1]);
  //threshold saturation
  //threshold(diff_s, diff_s, (saturation_threshold - hsv_means[1]), 255.0, THRESH_TOZERO);
  pow(diff_s, 2.0, diff_s);
  
  Mat maha_dist = ((1.0-sat_contri)/hs_variance[0])*diff_h + (sat_contri/hs_variance[1])*diff_s;
  sqrt(maha_dist, maha_dist);
  
  exp(-maha_dist, maha_dist);
  maha_dist = (1/0.02)*maha_dist;
  normalize(maha_dist, maha_dist, 0.0,1.0, NORM_MINMAX);
  
  //debug
  namedWindow("binary_image");
  imshow("binary_image", maha_dist);
  waitKey(0);
  

  //threshold
  //Mat bin_img(temp_h.rows, temp_h.cols, CV_32F);
  threshold(maha_dist, bin_img, hard_thresh, 1.0, THRESH_BINARY_INV);
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
      //double* curr_bin = bin_img.ptr<double>(i);
      
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


void new_comparison_mahalanobis(const Mat img_of_interest, const float *hsv_means, const float* hs_variance, Mat bin_img, bool isGreen)
{

  double hard_thresh, sigma, mean, s_thresh, v_thresh;
  
  int close_iterations = 1;
  Mat ioi_hsv, maha_dist;
  cvtColor(img_of_interest, ioi_hsv, CV_BGR2HSV);
  
  //split into different matrices the channels
  Mat temp_h, temp_s, temp_v;
  vector<Mat> arr_mats;
  split(ioi_hsv, arr_mats);
  
  temp_h = arr_mats[0];
  temp_s = arr_mats[1];
  temp_v = arr_mats[2];

  
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

  double max_value, min_value;
  minMaxLoc(maha_dist, &min_value, &max_value, NULL, NULL);
  cout<<"\nmax value:"<< max_value << "\nminvalue:"<<min_value<<endl;
  //normalize(maha_dist, maha_dist, 0.0,1.0, NORM_MINMAX);
  
  
  
  threshold(maha_dist, bin_img, hard_thresh, 1.0, THRESH_BINARY_INV);

  int struct_size = 3;
  Mat struc_elem = getStructuringElement(MORPH_ELLIPSE, Size(2*struct_size+1, 2*struct_size+1), Point (struct_size, struct_size));
  morphologyEx(bin_img, bin_img, MORPH_CLOSE, struc_elem, Point(-1,1), close_iterations);

  /*namedWindow("binary_image");
  imshow("binary_image", bin_img);
  waitKey(0);*/

}


//Finds color blob centroids using regular difference method(0) or 
//mahalanobis method(1) 
//false if no blob found
bool get_centroid(const Mat img_of_interest, const int method, const float color_means[], const float color_var[], 
                  const Mat color_cov_mat_inv, Point2f*  out_blob_centroid, double area_thresh,
                  tf::Transform& median_tf, bool isGreen)
{
  //convert to binary image
  Mat bin_img(img_of_interest.rows, img_of_interest.cols, CV_32F);
  
  if(method == 0)
    {
      //comparison_mahalanobis_diag(img_of_interest, color_means, color_var, bin_img);
      new_comparison_mahalanobis(img_of_interest, color_means, color_var, bin_img, isGreen);
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
  namedWindow("Binary Image", 1);
  imshow("Binary Image", bin_img);
  waitKey(0);*/

  if ((max_idx>-1) && (max_area>area_thresh))
    {
      Point2f cont_centr;
      float  cont_rad=0;
      minEnclosingCircle(contours[max_idx], cont_centr, cont_rad);
      //return the centroid computed
      *out_blob_centroid = cont_centr;

      accumulator_set<double, features<tag::median> > x_acc, y_acc, z_acc;
      if(cur_pc) {
        cout << endl << "h" << endl;
        cout << endl << cur_pc->points.size() << endl;
      }
      if(cur_pc && (fabs((cur_pc->header.stamp - last_img_time).toSec()) < 100.2)) {
        for(size_t i = 0; i < contours[max_idx].size(); i++) {
          PRGB pt = cur_pc->at(contours[max_idx][i].x, contours[max_idx][i].y);
          if(pt.x == pt.x && pt.y == pt.y && pt.z == pt.z) {
            x_acc(pt.x); y_acc(pt.y); z_acc(pt.z); 
          }
        }
        median_tf.setOrigin( tf::Vector3(median(x_acc), median(y_acc), median(z_acc)) );
        return true;
      } else {
        return false;
      }
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

void kin_rgb_fram_sub(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  last_img_time = msg->header.stamp;
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

void kin_pc_sub(sensor_msgs::PointCloud2::ConstPtr pc_msg) {
  pcl::fromROSMsg(*pc_msg, *cur_pc);
}


//MAIN
int main(int argc, char **argv)
{

  //Initialize ROS
  ros::init(argc, argv, "hand_tracker");
  ros::NodeHandle nh;
  image_transport::ImageTransport hand_tracking(nh);
  
  int frame_count = 0;

  string input_file_green = "src/green.txt", input_file_blue="src/blue.txt";

  //read blue variables
  float blue_means[2], blue_var[2], blue_hs_cov;
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
  cout<< green_cov_mat_inv;
   
  //Movie Window
  namedWindow( "Track Hands", 2 );
  int delay = 10;
  char c;
  
  //start looking at new images to find the colors tracked - loop here
  int frames_to_hold =10;
  Scalar cross_color1(0, 235, 0);
  Scalar cross_color2(0, 0, 245);
  
  image_transport::Subscriber kinect_rgb_sub = hand_tracking.subscribe
    ("kinect0/rgb/image_rect_color", frames_to_hold, &kin_rgb_fram_sub);

  ros::Subscriber kinect_pc_sub = nh.subscribe
    ("kinect0/depth_registered/points",  1, &kin_pc_sub);

  //  if(glob_img_of_interest.empty()){cout<<"Then say its empty..\n";}
 
  while(ros::ok())
    {
      ros::spinOnce();
      Mat img_of_interest = glob_img_of_interest;
      //smooth
      Size kernel_size(7,7);
      GaussianBlur(img_of_interest,img_of_interest,  kernel_size, 0.0);
      
      if(img_of_interest.empty()){cout<< "\n Its empty"<<endl;}
      else
	{
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
	  double blue_area_thresh = 15.0, green_area_thresh = 15.0;

	  //call function for centroid computation
    tf::Transform blue_median_tf, green_median_tf;
	  bool found_blue = get_centroid(img_of_interest, blue_method, blue_means, 
					 blue_var, blue_cov_mat_inv, &blue_centroid, blue_area_thresh, blue_median_tf, false);
	  bool found_green = get_centroid(img_of_interest, green_method, green_means, 
					  green_var, green_cov_mat_inv, &green_centroid, green_area_thresh, green_median_tf, true);
    static tf::TransformBroadcaster trans_broad;
    if(found_blue)
      trans_broad.sendTransform(tf::StampedTransform(blue_median_tf, cur_pc->header.stamp, cur_pc->header.frame_id, 
                                                     "purple_hand"));
    if(found_green)
      trans_broad.sendTransform(tf::StampedTransform(green_median_tf, cur_pc->header.stamp, cur_pc->header.frame_id, 
                                                     "green_hand"));

	  //debug - stop counting time
	  //t = ((double)getTickCount() - t)/getTickFrequency();
  

	  //debug- draw a cross to denote centroids

	  if(found_blue){draw_centroid(img_of_interest, blue_centroid, cross_color1);}
	  else{cout<<"\nCouldn't find blue hand\n";}
	  if(found_green){draw_centroid(img_of_interest, green_centroid, cross_color2);}
	  else{cout<<"\nCouldn't find green hand\n";}

	  //debug
	  //cout<<"I'm at least reaching\n";

	  imshow( "Track Hands", img_of_interest );
	  c = (char)cvWaitKey(delay);
	  if (c == 27) exit(-1);
	  
	  //debug - display execution time
	  //cout << "\nTimes passed in seconds: " << t << endl;
	  
	}
    }
  return 0;
}
