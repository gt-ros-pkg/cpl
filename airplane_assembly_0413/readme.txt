
1) Run Matlab inference code (only need to run once, it will automatically restart when finishing):
In Matlab, airplane_assembly_0413/matlab_inference
run ros_inference_loopie.m

2) Run inference ROS node:
rosrun airplane_assembly_inference_0313 inference_from_matlab.py

3) To view distributions:
rosrun airplane_assembly_inference_0313 distribution_viewer.py 























