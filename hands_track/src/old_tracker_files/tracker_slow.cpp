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
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

class handsTracker{

private:
  //declare frame globally
  ros::Time last_img_time;
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc;
  ros::NodeHandle nh;
  
  //kinect frame subscribe
  image_transport::Subscriber kinect_rgb_sub;
  //Point cloud subscribe	
  ros::Subscriber kinect_pc_sub ;

  //ASSUMING purple glove on right hand
  ros::Publisher purple_hand ;
  ros::Publisher green_hand ;
  ros::Publisher purple_hand_2d;
  ros::Publisher green_hand_2d;


public:

  //constructor
  handsTracker ()
  {
    //start looking at new images to find the colors tracked - loop here
    int frames_to_hold =1;
    image_transport::ImageTransport hand_tracking(nh);
    //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    //cur_pc = temp_pc;
    kinect_rgb_sub = hand_tracking.subscribe
      ("kinect0/rgb/image_rect_color", frames_to_hold, &handsTracker::kin_rgb_fram_sub, this);
    
    kinect_pc_sub = nh.subscribe
      ("kinect0/depth_registered/points",  1, &handsTracker::kin_pc_sub,this);

    //3d points
    purple_hand = nh.advertise<geometry_msgs::Point> ("right_hand", 0);
    green_hand = nh.advertise<geometry_msgs::Point> ("left_hand", 0);
    

  }
  
  
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





  void new_comparison_mahalanobis(const Mat img_of_interest, Mat bin_img, bool isGreen)
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
    //subtract(temp_h, temp_h, temp_h, temp_s < s_thresh);
    //subtract(temp_h, temp_h, temp_h, temp_v < v_thresh);
    subtract(temp_h, tempor, diff_h, noArray(), CV_32F);
    maha_dist = abs(diff_h);
  
    Mat new_thresh =  2*hard_thresh*Mat::ones(maha_dist.rows, maha_dist.cols, CV_32F);

    add(maha_dist, new_thresh, maha_dist, temp_s<s_thresh);
    add(maha_dist, new_thresh, maha_dist, temp_v<v_thresh);
  
    //debug
    /*cout << "mean " << mean << endl;
      cout << "sigma " << sigma << endl;
      cout << "s_thresh " << s_thresh << endl;*/

    //debug
    /*double max_value, min_value;
      minMaxLoc(maha_dist, &min_value, &max_value, NULL, NULL);
      cout<<"\nmax value:"<< max_value << "\nminvalue:"<<min_value<<endl;*/
    //normalize(maha_dist, maha_dist, 0.0,1.0, NORM_MINMAX);
  
  
    //threshold on the distance
    threshold(maha_dist, bin_img, hard_thresh, 1.0, THRESH_BINARY_INV);

    int struct_size = 3;
    Mat struc_elem = getStructuringElement(MORPH_ELLIPSE, Size(2*struct_size+1, 2*struct_size+1), Point (struct_size, struct_size));
    morphologyEx(bin_img, bin_img, MORPH_CLOSE, struc_elem, Point(-1,1), close_iterations);

    /*    namedWindow("binary_image");
    imshow("binary_image", bin_img);
    waitKey(0);*/

  }


  //Finds color blob centroids
  bool get_centroid(const Mat img_of_interest, Point2f*  out_blob_centroid, double area_thresh, bool isGreen, Point3f* centroid_3d)
  {
  
    //create holder for binary image
    Mat bin_img(img_of_interest.rows, img_of_interest.cols, CV_32F);
  
    new_comparison_mahalanobis(img_of_interest, bin_img, isGreen);
 

    /*//debug
      double max_value, min_value;
      minMaxLoc(bin_img, &min_value, &max_value, NULL, NULL);
      cout<<"\nmax value:"<< max_value << "\nminvalue:"<<min_value<<endl;*/
  
    //compute area and centroids
    vector<vector<Point> > contours;
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
	  for(size_t i = 0; i < contours[max_idx].size(); i++) 
	    {
	      PRGB pt = cur_pc->at(contours[max_idx][i].x, contours[max_idx][i].y);
	      if(pt.x == pt.x && pt.y == pt.y && pt.z == pt.z) {
		x_acc(pt.x); y_acc(pt.y); z_acc(pt.z); 
	      }
	    }
        
	  Point3f temp_3d;
	  temp_3d.x=median(x_acc); temp_3d.y=median(y_acc), temp_3d.z=median(z_acc);
	  *centroid_3d = temp_3d;
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


  //publish 3d position of centroid using input publisher
  //it publishes (inf,inf,inf) in case hand was not found
  void publish_centroid(Point3f centroid, bool found, ros::Publisher pub)
  {
  
    geometry_msgs::Point msg;
  
    if(found)
      {
 	msg.x=centroid.x;msg.y=centroid.y;msg.z=centroid.z;
      }else
      {
	double inf = std::numeric_limits<double>::infinity();
	msg.x=inf; msg.y=inf; msg.z=inf;
      }
  
    //publish message
    pub.publish(msg);
  
    //debug
    ROS_INFO("Transmitted a hand %d");

  }


  void frame_process(Mat img_of_interest)
  {

   
    //debug - Movie Window
    namedWindow( "Track Hands", 2 );
    int delay = 5;
    char c;
	
    Scalar cross_color1(0, 235, 0);
    Scalar cross_color2(0, 0, 245);
  
	
    if(!img_of_interest.empty())
      {
	//debug
	//debug - start counting time
	//double t = (double)getTickCount();
  
	//parameters for comparison
	int blue_method = 0, green_method=0;
	double blue_threshold, green_threshold;
	Point2f blue_centroid, green_centroid;
	Point3f blue_centroid_3, green_centroid_3;
	  
	//currently only area used
	double blue_area_thresh = 15.0, green_area_thresh = 15.0;

	//call function for centroid computation
	bool found_blue = get_centroid(img_of_interest, &blue_centroid, blue_area_thresh, false, &blue_centroid_3);
	bool found_green = get_centroid(img_of_interest, &green_centroid, green_area_thresh,true, &green_centroid_3);
	  

	publish_centroid(blue_centroid_3, found_blue, purple_hand);
	publish_centroid(green_centroid_3, found_green, green_hand);
	//debug - stop counting time
	//t = ((double)getTickCount() - t)/getTickFrequency();
  

	//debug- draw a cross to denote centroids
	if(found_blue){draw_centroid(img_of_interest, blue_centroid, cross_color1);}
	else{cout<<"\nCouldn't find blue hand\n";}
	if(found_green){draw_centroid(img_of_interest, green_centroid, cross_color2);}
	else{cout<<"\nCouldn't find green hand\n";}
	//debug
	/*imshow( "Track Hands", img_of_interest );
	c = (char)cvWaitKey(delay);
	if (c == 27) exit(-1);*/
	  
	//debug - display execution time
	//cout << "\nTimes passed in seconds: " << t << endl;
	  
      }

  }

  //callback - grabs frame from kinect and processes it
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

    Mat img_of_interest = cv_ptr->image;
    //debug
    //imshow("show", glob_img_of_interest);
    //waitKey(0);
    
    //process
    frame_process(img_of_interest);
  }

  //callback - grabs point cloud
  void kin_pc_sub(sensor_msgs::PointCloud2::ConstPtr pc_msg) {
    /*if ((pc_msg->width * pc_msg->height) == 0)
         return; //return if the cloud is not dense!
       try
       {
         pcl::fromROSMsg (*pc_msg, *cur_pc); //convert the cloud
       }
       catch (std::runtime_error e)
       {
         ROS_ERROR_STREAM("Error in converting cloud to image message: "
			    << e.what());
			    }*/    

    pcl::fromROSMsg(*pc_msg, *cur_pc);
  }

};


//MAIN
int main(int argc, char **argv)
{

  
  ros::init(argc, argv, "hand_tracker"); 
  handsTracker begin_track;

  
  ros::spin();

      

  return 0;
}

