cmake_minimum_required(VERSION 2.8.3)
project(planetary_slam)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 17)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  tf
  visualization_msgs
)

set(Eigen3_DIR /usr/lib/cmake/eigen3)
find_package(Eigen3 3.3.4 REQUIRED)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Glog REQUIRED)
find_package(libpointmatcher REQUIRED)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 手动指定libnabo路径
set(LIBNABO_INCLUDE_DIRS /usr/local/include)
set(LIBNABO_LIBRARY_DIRS /usr/local/lib)
set(LIBNABO_LIBRARIES nabo)

set(teaserpp_DIR "/usr/local/lib/cmake/teaserpp")
find_package(teaserpp REQUIRED)

# 添加 CSF 库和头文件
set(CSF_INCLUDE_DIR "/usr/local/include")  # CSF 安装路径
set(CSF_LIBRARY "/usr/local/lib/libCSF.a") # CSF 静态库或动态库路径

# 将 CSF 包含路径添加到 include_directories
include_directories(${CSF_INCLUDE_DIR})

set(Boost_DIR "/home/gaoyang/miniconda3/pkgs/boost-cpp-1.72.0-he72f1d9_7/lib/cmake/Boost-1.72.0")
find_package(Boost REQUIRED NO_MODULE)


include_directories(
  ${PROJECT_SOURCE_DIR}/include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${LIBNABO_INCLUDE_DIRS}
  ${libpointmatcher_INCLUDE_DIRS}
  ${teaserpp_INCLUDE_DIRS}
  /usr/local/include
)

link_directories(${LIBNABO_LIBRARY_DIRS} ${libpointmatcher_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL Ceres Glog
  INCLUDE_DIRS include
)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(nlohmann_json)


add_library(rangeimg_lib src/range_image.cpp)
# target_link_libraries(rangeimg_lib )

add_library(common_lib src/common.cpp)
target_link_libraries(common_lib nlohmann_json::nlohmann_json)

add_library(saver_lib src/saver.cpp)
target_link_libraries(saver_lib PUBLIC common_lib pointmatcher ${catkin_LIBRARIES} ${PCL_LIBRARIES} OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)

add_library(solver_lib src/solver.cpp)
target_link_libraries(solver_lib PUBLIC ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${Boost_INCLUDE_DIRS} nlohmann_json::nlohmann_json teaserpp::teaser_registration teaserpp::teaser_io)

add_executable(scan_registration src/scan_registration.cpp)
target_link_libraries(scan_registration saver_lib rangeimg_lib ${CERES_LIBRARIES} ${CSF_LIBRARY})

add_executable(laser_odometry src/laser_odometry.cpp src/imls_icp.cpp)
target_link_libraries(laser_odometry saver_lib solver_lib ${GLOG_LIBRARIES} ${LIBNABO_LIBRARIES})

