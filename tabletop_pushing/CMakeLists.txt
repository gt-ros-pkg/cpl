cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)
# set(ROS_BUILD_TYPE Debug)

# Build actions
rosbuild_find_ros_package(actionlib_msgs)
include(${actionlib_msgs_PACKAGE_PATH}/cmake/actionbuild.cmake)
genaction()

# add local path for finding LibSVM
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/Modules/)
message("module path" ${CMAKE_MODULE_PATH})

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

# Find external libraries
find_package(OpenCV REQUIRED)
find_package(LibSVM)

rosbuild_add_library(graphcut include/tabletop_pushing/extern/graphcut/graph.cpp include/tabletop_pushing/extern/graphcut/maxflow.cpp include/tabletop_pushing/extern/graphcut/LinkedBlockList.cpp include/tabletop_pushing/extern/graphcut/GCoptimization.cpp)

rosbuild_add_library(point_cloud_segmentation include/tabletop_pushing/point_cloud_segmentation.h src/point_cloud_segmentation.cpp)
target_link_libraries(point_cloud_segmentation ${OpenCV_LIBRARIES})

rosbuild_add_library(object_tracker_25d include/tabletop_pushing/object_tracker_25d.h src/object_tracker_25d.cpp)
target_link_libraries(object_tracker_25d ${OpenCV_LIBRARIES})

rosbuild_add_library(shape_features include/tabletop_pushing/shape_features.h src/shape_features.cpp)
target_link_libraries(shape_features ${OpenCV_LIBRARIES})

rosbuild_add_library(arm_obj_segmentation include/tabletop_pushing/arm_obj_segmentation.h src/arm_obj_segmentation.cpp)
target_link_libraries(arm_obj_segmentation graphcut)
target_link_libraries(arm_obj_segmentation ${OpenCV_LIBRARIES})

rosbuild_add_executable(tabletop_pushing_perception_node src/tabletop_pushing_perception_node.cpp)
target_link_libraries(tabletop_pushing_perception_node point_cloud_segmentation)
target_link_libraries(tabletop_pushing_perception_node shape_features)
target_link_libraries(tabletop_pushing_perception_node object_tracker_25d)
target_link_libraries(tabletop_pushing_perception_node arm_obj_segmentation)
target_link_libraries(tabletop_pushing_perception_node ${OpenCV_LIBRARIES})
rosbuild_link_boost(tabletop_pushing_perception_node signals)
rosbuild_link_boost(tabletop_pushing_perception_node signals)
target_link_libraries(tabletop_pushing_perception_node ${LIBSVM_LIBRARIES})

# rosbuild_add_executable(test_svm src/test_svm.cpp)
# target_link_libraries(test_svm ${LIBSVM_LIBRARIES})
# target_link_libraries(test_svm shape_features)

rosbuild_add_executable(object_singulation_node src/object_singulation_node.cpp)
target_link_libraries(object_singulation_node point_cloud_segmentation)
target_link_libraries(object_singulation_node ${OpenCV_LIBRARIES})
rosbuild_link_boost(object_singulation_node signals)

rosbuild_add_executable(save_kinect_data src/save_kinect_images.cpp)
target_link_libraries(save_kinect_data point_cloud_segmentation)
target_link_libraries(save_kinect_data ${OpenCV_LIBRARIES})
rosbuild_link_boost(save_kinect_data signals)

rosbuild_add_executable(extract_shape_features src/extract_shape_features.cpp)
target_link_libraries(extract_shape_features shape_features)
target_link_libraries(extract_shape_features ${LIBSVM_LIBRARIES})

# rosbuild_add_executable(pca_test src/pca_test.cpp)