#ifndef SAVER_H
#define SAVER_H

#include <fstream>
#include <iomanip>
#include <string>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pointmatcher/PointMatcher.h>

#include "common.h"


void savePointCloudToTxt(const typename pcl::PointCloud<PointType>::Ptr& cloud, const std::string& filename);
void savePoseToFile(const Eigen::Matrix4d& pose, const std::string& filename, const std::string& timestamp);
void saveMarkerToFile(const visualization_msgs::Marker& marker, const std::string& filename);
void saveThresholdFile(const std::string& filename, double number);
void saveCloudFeaturesAndDescriptors(const DP& dp_cloud, const std::string& filename);
void saveMatchedPointsToFile(
    const std::vector<Eigen::Vector3d>& source_cloud,
    const std::vector<Eigen::Vector3d>& matched_cloud,
    const std::string& filename);

sensor_msgs::PointCloud2 libPointMatcherToRosMsg(const DP& dpCloud, 
                                                 const std::string& frame_id, const ros::Time& stamp);
DP rosMsgToLibPointMatcherCloud(const sensor_msgs::PointCloud2& rosMsg);

void publishPointCloud(
    const pcl::PointCloud<PointType>::Ptr& cloud,
    ros::Publisher& publisher,
    const ros::Time& stamp,
    const std::string& frame_id = "/camera_init");
void publishDataPointsAsPointCloud2(
    const DP& dp_cloud, 
    const std::string& frame_id, 
    const ros::Publisher& publisher, 
    const ros::Time& stamp);
void pubPath(
    const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation,
    nav_msgs::Path& path,
    ros::Publisher& mcu_path_pub_,
    const ros::Time& stamp,
    const std::string& frame_id = "/camera_init");
void visualizePCAFeatures(
    const pcl::PointCloud<PointType>::Ptr& laserCloud,
    visualization_msgs::Marker& pcaMarker);

#endif // SAVER_H
