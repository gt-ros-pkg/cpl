#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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

Scalar cross_color1(0, 235, 0);
Scalar cross_color2(0, 0, 245);


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
    
    subtract(temp_h, tempor, diff_h, noArray(), CV_32F);
    maha_dist = abs(diff_h);
  
    Mat new_thresh =  2*hard_thresh*Mat::ones(maha_dist.rows, maha_dist.cols, CV_32F);

    add(maha_dist, new_thresh, maha_dist, temp_s<s_thresh);
    add(maha_dist, new_thresh, maha_dist, temp_v<v_thresh);
  
    
  
    //threshold on the distance
    threshold(maha_dist, bin_img, hard_thresh, 1.0, THRESH_BINARY_INV);

    int struct_size = 3;
    Mat struc_elem = getStructuringElement(MORPH_ELLIPSE, Size(2*struct_size+1, 2*struct_size+1), Point (struct_size, struct_size));
    morphologyEx(bin_img, bin_img, MORPH_CLOSE, struc_elem, Point(-1,1), close_iterations);

    /*    namedWindow("binary_image");
    imshow("binary_image", bin_img);
    waitKey(0);*/

}

void FindBlobs(const Mat &binary, vector < vector<Point2i> > &blobs, int* max_idx_out, int* max_area_out)
{
    blobs.clear();
    int max_area =0;
    int max_label =-1;
    vector<int> label_areas;
    label_areas.clear();
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32FC1);
    
    /*namedWindow("to label",1);
    imshow("to label", label_image);
    waitKey(0);*/
    
    float label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < binary.rows; y++) {
      for(int x=0; x < binary.cols; x++) {
	
	if(label_image.at<float>(y,x) != float(1)) {
	  continue;
	}

	Rect rect;
	floodFill(label_image, Point(x,y), Scalar(label_count), &rect);
	
	vector <Point2i> blob;
	blob.clear();
	int label_area=0;
	
	for(int i=rect.y; i < (rect.y+rect.height); i++) {
	  for(int j=rect.x; j < (rect.x+rect.width); j++) {
	    if(int(label_image.at<float>(i,j)) != label_count) {continue;}
	    blob.push_back(Point2i(j,i));
	    // cout<<"not here then?";
	    //waitKey(0);
	    label_area++;
	  }
	}

	label_areas.push_back(label_area);
	blobs.push_back(blob);
	
	//cout<<"HERE?"<<endl<<label_area;
	//check-max
	if (label_area>max_area)
	  {
	    //cout<<"ever here?"<<endl;
	    max_area=label_area; 
	    max_label=label_count-2;
	  }
	
	label_count++;
      }
    }
    cout<<"label_count"<<label_count<<endl;
    *max_idx_out = max_label;
    *max_area_out = max_area;
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
    vector<vector<Point2i> > contours;
    Mat new_bin(bin_img.rows, bin_img.cols, CV_8U);
    bin_img.convertTo(new_bin, CV_8U, 1);
    
    //debug - visualize biggest contour
    Mat drew_contour = Mat::zeros(new_bin.size(), CV_8UC3);
    //new_bin.copyTo(drew_contour);

    vector<vector<Point2i> >  blobs;
    int max_area, max_idx; 
    cout<<"starrt here"<<endl;
    FindBlobs(new_bin, blobs, &max_idx,  &max_area);
    cout<<"here?";
    /*//debug - show image
      namedWindow("Binary Image", 1);
      imshow("Binary Image", bin_img);
      waitKey(0);*/
    
    cout<<"max-area"<<max_area<<"\nmax id"<<max_idx<<endl;
    if ((max_idx>-1) && (max_area>area_thresh))
      {

	accumulator_set<int, features<tag::median> > x_acc, y_acc, z_acc;
	/*if(cur_pc) {
	  cout << endl << "h" << endl;
	  cout << endl << cur_pc->points.size() << endl;
	  }*/

	//	if(cur_pc && (fabs((cur_pc->header.stamp - last_img_time).toSec()) < 100.2)) 
	cout<<"total soze:"<<blobs.size()<<endl;
	for(size_t i = 0; i < blobs[max_idx].size(); i++) 
	    {
	      //     PRGB pt = cur_pc->at(contours[max_idx][i].x, contours[max_idx][i].y);
	      //if(pt.x == pt.x && pt.y == pt.y && pt.z == pt.z) {
	      cout<<"once";
	      x_acc(blobs[max_idx][i].x); y_acc(blobs[max_idx][i].y); 
	      
	      
		// }
	    }
	cout<<"reach here??";
	out_blob_centroid->x = float(median(x_acc)); out_blob_centroid->y = float(median(y_acc));
	  
	
	      Vec3b color( rand()&255, rand()&255, rand()&255);
	      for (size_t j=0; j<blobs[max_idx].size(); j++)
		{
		  
		  drew_contour.at<Vec3b>(blobs[max_idx][j].x, blobs[max_idx][j].y) = color;

		}
		
	  //Scalar color( rand()&255, rand()&255, rand()&255);
	  //floodFill(drew_contour, contours[max_idx][0], color);
	  namedWindow("Each Contour",1);
	  namedWindow("bin image",1);
	  imshow("Each Contour", drew_contour);
	  imshow("bin image", new_bin);
	  waitKey(0);
	  
	  
	  Point3f temp_3d;
	  temp_3d.x=median(x_acc); temp_3d.y=median(y_acc), temp_3d.z=median(z_acc);
	  *centroid_3d = temp_3d;
	  return true;
	  
      }

    else{
      //cout<<"\nNot found blob\n";
      return false;}
       
  }
void draw_centroid(Mat img_of_interest, Point2i in_centroid, const Scalar cross_color)
{

  int line_thickness = 5;
  Point2i centroid_1, centroid_2;
  centroid_1 = centroid_2 = in_centroid;
  centroid_1.x = std::max(0, in_centroid.x-10);
  centroid_2.x = std::min(img_of_interest.cols, in_centroid.x+10);
  line(img_of_interest, centroid_1, centroid_2, cross_color, line_thickness);

  centroid_1 = centroid_2 = in_centroid;
  centroid_1.y = std::max(0, in_centroid.y-10);
  centroid_2.y = std::min(img_of_interest.cols, in_centroid.y+10);
  line(img_of_interest, centroid_1, centroid_2, cross_color, line_thickness);

}


void frame_process(Mat img_of_interest)
{
	
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
	  

      //      publish_centroid(blue_centroid_3, found_blue, purple_hand);
      //publish_centroid(green_centroid_3, found_green, green_hand);
      //debug - stop counting time
      //t = ((double)getTickCount() - t)/getTickFrequency();
  

      //debug- draw a cross to denote centroids
      if(found_blue){draw_centroid(img_of_interest, blue_centroid, cross_color1);}
      else{cout<<"\nCouldn't find blue hand\n";}
      if(found_green){draw_centroid(img_of_interest, green_centroid, cross_color2);}
      else{cout<<"\nCouldn't find green hand\n";}
      //debug
      imshow( "Track Hands", img_of_interest );
      int delay=20;
      waitKey(0);
	char c = (char)cvWaitKey(delay);
	if (c == 27) exit(-1);
	  
      //debug - display execution time
      //cout << "\nTimes passed in seconds: " << t << endl;
	  
    }
}


//MAIN
int main(int argc, char **argv)
{
  
  double frame_count = 0;
  double total_time_counter;
  VideoCapture vid_of_interest("kinect_stuff.avi");
  if (!vid_of_interest.isOpened())
    {
      cout<< "\nVideo Open Unsuccessful!"<<endl;
      exit(-1);
    }
  else
    {
      
      Mat img_of_interest;
      vid_of_interest >> img_of_interest;
      
      
      
      while(!img_of_interest.empty())
	{
	  double t = (double)getTickCount();
	  vid_of_interest >> img_of_interest;

	  frame_count++;
	  //debug - start counting time
	  // double t = (double)getTickCount();
      
	  frame_process(img_of_interest);
	  //debug - stop counting time
	  t = ((double)getTickCount() - t)/getTickFrequency();
	  total_time_counter += t; 
	  cout<< t<<endl; 
	  
	  
	}
    }

  cout<< "Average time: " <<  (total_time_counter/frame_count);
  return 0;
}


