#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cv.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <math.h>

using namespace cv;
using namespace std;

boost::mt19937 rng;
boost::normal_distribution<> nd(0.0, 1.0);
boost::variate_generator<boost::mt19937&, 
			 boost::normal_distribution<> > var_nor(rng, nd);



//TODO: 1)test it out on ros; 2) tune parameters; 3)look at point cloud
// 4) plot x,y,z positions; 5)try including more into state var
// 6) profile for speed; 7)compare with kalman; 8)compare with only z filt

class particleFilter{

private:

  struct particle{
    Point3f location;
    double weight; 
  };  

  int NO_OF_PARTICLES;
  vector<particle> PARTICLES;
  vector<particle> PARTICLES_PRED;
  double MAX_PART_WT;

  double state_transition_std;
  double mse_std;
  
  //stores last seen evidence
  Point3f last_obs;
  
public:
  
  //default
  particleFilter(){}
  
  //Initialize - number of particles, sample uniformly at random in defined 
  //2d bounding box with replacement
  void general_init(int num_part, double transit_std, double mse_std_in)
  {
    //set std dev of gaussian similarity function
    mse_std = mse_std_in;
    
    NO_OF_PARTICLES = num_part;

    //std dev for state transition gaussian noise
    state_transition_std = transit_std;    
    //create particles equally weighted, positioned at origin
    particle temp;
    Point3f temp_loc(0.0, 0.0, 0.0);

    temp.location = temp_loc;
    temp.weight = 1.0/num_part;
    MAX_PART_WT = temp.weight;
    
    PARTICLES.clear();

    //add each particle into particles
    for (int i=0; i<NO_OF_PARTICLES;i++)
      {
	/*//estimate x
	temp.location.x = (upper_right.x-bottom_left.x) * 
	  (double(rand())/RAND_MAX) + bottom_left.x; 
	//estimate y
	temp.location.y = (upper_right.y-bottom_left.y) * 
	  (double(rand())/RAND_MAX) + bottom_left.y;
	//estimate z
	temp.location.z = (z_width) * 
	(double(rand())/RAND_MAX) + (mean_z + (z_width/2));*/
	
	
	PARTICLES.push_back(temp);
      }
      

  }


  /*//Initialize with a gaussian(1d)
  //TODO: 2d or use a GMM
  particleFilter(double mean, double variance){}*/
  
  //initialize particles around point passed as arg
  void initial_arnd_mean(Point3f mean_pt, double dev)
  {
    //debug
    //    cout<<"MEAN = "<< mean_pt << "std_dev = "<< dev;
    
    Point3f temp_loc = mean_pt;

    //debug
    // cout<<"MEAN = "<< temp_loc << "std_dev = "<< dev<<" Random:"<<var_nor();

    for (vector<particle>::iterator it = PARTICLES.begin(); 
	 it != PARTICLES.end(); ++it)
      {
	//add gaussian noise with mean 0 and sigma dev
	temp_loc.x += dev*var_nor();
	temp_loc.y += dev*var_nor();
	temp_loc.z += dev*var_nor();
	(*it).location = temp_loc;
	(*it).weight = 1.0/double(NO_OF_PARTICLES);
	
	//debug
	//	cout<<"\n x y z: "<<temp_loc.x<<' '<<temp_loc.y<<' '<<temp_loc.z<<endl; 
      }
    MAX_PART_WT = 1.0/ double(NO_OF_PARTICLES);
    
    /*//debug
    cout<<"\ninitialize\n";
    disp_particles();*/
  }

  //Randomly sample a particle from the weighted distribution
  particle sampleParticle()
  {
    bool accept = false;
    particle temp;
    int rand_idx;
    double rand_likelihood;
      
    //accept-reject sampling using weights
    while(!accept)
      {
	rand_idx = rand()%NO_OF_PARTICLES;
	rand_likelihood = MAX_PART_WT * (double(rand())/RAND_MAX);
	temp = PARTICLES[rand_idx];
	if (temp.weight>=rand_likelihood){accept = true;}
      }
    
    return temp;
  }
  
  //add gaussian noise to all 3 dimensions
  Point3f dynamic_perturb(Point3f part_pos)
  {
    part_pos.x += state_transition_std*var_nor();
    part_pos.y += state_transition_std*var_nor();
    part_pos.z += state_transition_std*var_nor();

    return part_pos;
  }
  
  //update particle weights given obs by keeping it proportional 
  //to the negative exponent of the euclidean distance 
  void update(Point3f observation, bool obs_seen)
  {
    particle new_particle;
    double dist_from_obs;
    Point3f temp_loc;
    double norm_factor=0.0;
    
    //clear prediction particle vector so can push newly predicted samples
    PARTICLES_PRED.clear();
    
    //in case new observation not made, substitute with last
    if(!obs_seen){observation = last_obs;}
    else{ last_obs = observation;}

    for (int i=0; i<NO_OF_PARTICLES; i++)
      {
	//sample
	new_particle = sampleParticle();
	//transition state next time-step
	new_particle.location = dynamic_perturb(new_particle.location);
	//euclidean distance from observation
	temp_loc = new_particle.location;
	dist_from_obs = sqrt(pow(temp_loc.x-observation.x,2) + 
			     pow(temp_loc.y-observation.y,2) + 
			     pow(temp_loc.z-observation.z,2));
	//re-weight by squared negative exponent of dist
	//new_particle.weight = exp(-dist_from_obs/(2*pow(mse_std,2)));
	new_particle.weight = mse_std - dist_from_obs;
	
	//debug
	/*cout<<"\nDist = "<<dist_from_obs<<
	  "   similarity="<<new_particle.weight<<endl;
	  waitKey(0);*/


	/*	//debug
	if (isnan(new_particle.weight)){cout<<"stop here"; exit(-1);}
	if(new_particle.weight==0.0){cout<<"\n weights becoming zero ";
	cout<<"  distance="<<dist_from_obs<<"  this="<<(-dist_from_obs/(2*pow(mse_std,2)));exit(-1);}*/
	
	norm_factor += new_particle.weight;
	//add it to the list of particles predicted
	PARTICLES_PRED.push_back(new_particle);
      }
    
    //debug
    cout<<"\nDONE!!!!!!\n";
    if (norm_factor<=0){cout<<"\nNormalizing Factor not positive."; exit(-1);}
    
    //normalize weights    
    MAX_PART_WT = 0.0;
    for (vector<particle>::iterator it = PARTICLES_PRED.begin(); 
	 it != PARTICLES_PRED.end(); ++it)
      {
	(*it).weight /= norm_factor;

	/*//debug
	if(isnan((*it).weight)){cout<<"\nnorm_factor="<<norm_factor<<endl;}
	if ((*it).weight > MAX_PART_WT){MAX_PART_WT = (*it).weight;}*/
      }
    
    //assign to current particle vector
    PARTICLES.clear();
    PARTICLES = PARTICLES_PRED;
  }
  
  
  //take weighted mean of the particles locations predicted
  //and return this as a measurement
  Point3f predict_pos_wt_mean()
  {
    Point3f mean_point(0.0, 0.0, 0.0);
    for (vector<particle>::iterator it = PARTICLES.begin();
	 it != PARTICLES.end(); ++it)
      {
	mean_point.x += (((*it).location.x)*((*it).weight));
	mean_point.y += (((*it).location.y)*((*it).weight));
	mean_point.z += (((*it).location.z)*((*it).weight));
      }

    //debug
    if (isnan(mean_point.x) || isnan(mean_point.y) || isnan(mean_point.z))
      {
	cout<< "The corrected mean has NaN"<<endl;
	disp_particles();
	exit(-1);
      }

    
    return mean_point;
  }
  
  //debug
  //display particle positions and weights
  void disp_particles()
  {
    int counter = 1;
    for (vector<particle>::iterator it = PARTICLES.begin();
	 it != PARTICLES.end(); ++it)
      {
	cout<<endl<< "Particle"<<counter;
	cout<< "Location  "<<(*it).location;
	cout<< "Weight" << (*it).weight;
	++counter;
      }
  }
};


class handPosFilter
{
private:
  ros::NodeHandle nh;
  ros::Subscriber left_hand_sub ;
  ros::Subscriber right_hand_sub ;
  
  ros::Publisher right_hand_pub;
  ros::Publisher left_hand_pub;

  particleFilter lh_filter;
  particleFilter rh_filter;

  //denote the filter is yet to be initialized
  bool first_time_left;
  bool first_time_right;

  //std-dev for initial particle distribution
  double initial_part_stddev;
  
public:

  handPosFilter ()
  {
    int frames_to_hold =2;

    left_hand_sub = nh.subscribe
      ("/left_hand", frames_to_hold, &handPosFilter::grab_lefty, this);
    
    right_hand_sub = nh.subscribe
      ("/right_hand", frames_to_hold, &handPosFilter::grab_righty, this);

    //3d points
    left_hand_pub = nh.advertise<geometry_msgs::PoseStamped> ("left_hand_filt", frames_to_hold);
    right_hand_pub = nh.advertise<geometry_msgs::PoseStamped> ("right_hand_filt", frames_to_hold);

    first_time_left = true;
    first_time_right = true;

    int number_particles = 500;
    double part_transition_stddev = 0.02;
    double mse_stddev = 1.5;
    initial_part_stddev = 0.01;

    lh_filter.general_init(number_particles, part_transition_stddev, 
			   mse_stddev);
    rh_filter.general_init(number_particles, part_transition_stddev, 
			   mse_stddev);
  }

  void geo_pt_to_ocv_pt(geometry_msgs::Point geo_pt, Point3f* ocv_pt)
  {
    ocv_pt->x = geo_pt.x;
    ocv_pt->y = geo_pt.y;
    ocv_pt->z = geo_pt.z;
  }

  void ocv_pt_to_geo_pt(Point3f ocv_pt, geometry_msgs::Point* geo_pt)
  {
    geo_pt->x = ocv_pt.x;
    geo_pt->y = ocv_pt.y;
    geo_pt->z = ocv_pt.z;
  }

  void grab_lefty(geometry_msgs::PoseStamped hand_pos)
  {
    if (first_time_left)
      {

	if ((hand_pos.pose.position.x==numeric_limits<double>::infinity()) ||
	    (hand_pos.pose.position.y==numeric_limits<double>::infinity()) ||
	    (hand_pos.pose.position.z==numeric_limits<double>::infinity()))
	  {
	    left_hand_pub.publish(hand_pos);
	  }else
	  {
	    first_time_left = false;
	    
	    Point3f hand_filt;
	    geo_pt_to_ocv_pt(hand_pos.pose.position, &hand_filt);
	    //debug
	    cout<<"\nFirst left- before filt"<<hand_filt;
	    
	    lh_filter.initial_arnd_mean(hand_filt, initial_part_stddev);
	    lh_filter.update(hand_filt, true);
	    hand_filt = lh_filter.predict_pos_wt_mean();
	    //debug
	    cout<<"\nFirst left- after filt"<<hand_filt<<endl;

	    ocv_pt_to_geo_pt(hand_filt, &hand_pos.pose.position) ;
	    left_hand_pub.publish(hand_pos);
	  }
      }
    else
      {
	bool obs_seen =true;
	if ((hand_pos.pose.position.x==numeric_limits<double>::infinity()) ||
	    (hand_pos.pose.position.y==numeric_limits<double>::infinity()) ||
	    (hand_pos.pose.position.z==numeric_limits<double>::infinity()))
	  { obs_seen = false;}
	Point3f hand_filt;
	geo_pt_to_ocv_pt(hand_pos.pose.position, &hand_filt);
	lh_filter.update(hand_filt, obs_seen);
	hand_filt = lh_filter.predict_pos_wt_mean();
	ocv_pt_to_geo_pt(hand_filt, &hand_pos.pose.position) ;
	left_hand_pub.publish(hand_pos);
      }
  }

  void grab_righty(geometry_msgs::PoseStamped hand_pos)
  {
    if (first_time_right)
      {

	if ((hand_pos.pose.position.x==numeric_limits<double>::infinity()) ||
	    (hand_pos.pose.position.y==numeric_limits<double>::infinity()) ||
	    (hand_pos.pose.position.z==numeric_limits<double>::infinity()))
	  {
	    right_hand_pub.publish(hand_pos);
	  }else
	  {
	    first_time_right = false;
	    Point3f hand_filt;
	    geo_pt_to_ocv_pt(hand_pos.pose.position, &hand_filt);
	    
	    //debug
	    cout<<"\nFirst right- before filt"<<hand_filt;
	    
	    rh_filter.initial_arnd_mean(hand_filt, initial_part_stddev);
	    rh_filter.update(hand_filt, true);
	    hand_filt = rh_filter.predict_pos_wt_mean();
	    ocv_pt_to_geo_pt(hand_filt, &hand_pos.pose.position) ;
	    
	    //debug
	    cout<<"\nFirst right- before filt"<<hand_filt;
	    
	    right_hand_pub.publish(hand_pos);
	  }
      }
    else
      {
	bool obs_seen =true;
	if ((hand_pos.pose.position.x==numeric_limits<double>::infinity()) ||
	    (hand_pos.pose.position.y==numeric_limits<double>::infinity()) ||
	    (hand_pos.pose.position.z==numeric_limits<double>::infinity()))
	  { obs_seen = false;}
	Point3f hand_filt;
	geo_pt_to_ocv_pt(hand_pos.pose.position, &hand_filt);
	rh_filter.update(hand_filt, obs_seen);
	hand_filt = rh_filter.predict_pos_wt_mean();
	ocv_pt_to_geo_pt(hand_filt, &hand_pos.pose.position) ;
	right_hand_pub.publish(hand_pos);
      }
  }

  
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "part_filt_hands"); 
  handPosFilter begin_track;

  ros::spin();

  return 0;
}
