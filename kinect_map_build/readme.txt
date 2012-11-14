Name: Kelsey Hawkins
GTID: khawkins3

Dataset can be found at 
http://dl.kelseyhawkins.com/files/cpl_01_preprocess.bag

I created this dataset by holding up a Kinect and continuously panning around the room.  
Transformations are performed in order of collection, downsampling by 10 so that
frames are captured at approximately 3 Hz.  I have not created a spanning tree, but
have provided the images in order, so the edges should connect subsequent images in
the list.  Only a N=10 subset of the dataset was used. Attached is a video, 
dataset_viz.ogv, which visualizes the dataset I captured.

The OpenCV SURF feature detection produced an average of 1023 keypoints per image.

The libraries used are entirely built into ROS, and include PCL for point cloud capture
and manipulation, OpenCV for image processing, Eigen for linear algebra, and
rosbag/rosmsg for data storage/marshalling. I also used a library, not included,
which converts ROS bags into .ply files.

The distance threshold I used for inlier detection was 30 cm and ran 400 iterations per RANSAC procedure.
rosrun kinect_map_build kinect_mapping ../data/cpl_01_preprocess.bag cpl_01_pre_out2.bag _start:=70 _step:=10 _dist_thresh:=0.2 _sample_size:=4 _num_iters:=400
