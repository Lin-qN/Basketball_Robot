cmake_minimum_required(VERSION 3.0.2)
project(yolov5)

add_compile_options(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  std_msgs
  message_generation
)
find_package(CUDA REQUIRED)
find_package(OpenCV 4 REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

add_message_files(
  FILES
  result.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES yolov5
  CATKIN_DEPENDS config roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ../../devel/include
)

#CUDA
include_directories(/usr/local/cuda-11.3/include)
link_directories(/usr/local/cuda-11.3/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/home/lin/TensorRT-8.0.0.3/include)
link_directories(/home/lin/TensorRT-8.0.0.3/lib/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED src/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)


include_directories(OpenCV_INCLUDE_DIRS)


cuda_add_executable(yolov5_node src/calibrator.cpp src/yolov5.cpp src/preprocess.cu)
target_link_libraries(yolov5_node
  ${catkin_LIBRARIES}
  nvinfer
  cudart
  myplugins
  ${OpenCV_LIBS}
)

add_definitions(-O2 -pthread)
add_definitions(-DPROJECT_PATH="${PROJECT_SOURCE_DIR}")
