cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#include_directories(/usr/include/vtk-5.8)
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(EXECUTABLES test_surf extract_img preprocess_bag kinect_mapping)
foreach(executable ${EXECUTABLES})
    rosbuild_add_executable(${executable} src/${executable}.cpp)
    target_link_libraries(${executable} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})
endforeach(executable)
