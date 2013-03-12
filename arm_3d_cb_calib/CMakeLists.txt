cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


find_package(GTSAM)

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

rosbuild_add_library(${PROJECT_NAME} src/calib_3d_cbs.cpp)
rosbuild_add_executable(test_cb_calib src/test_cb_calib.cpp)
target_link_libraries(test_cb_calib ${PROJECT_NAME} gtsam-static)
rosbuild_add_executable(find_cb_calib src/find_cb_calib.cpp)
target_link_libraries(find_cb_calib ${PROJECT_NAME} gtsam-static)
