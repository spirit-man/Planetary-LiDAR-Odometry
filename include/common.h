#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <vector>
#include <iomanip>
#include <omp.h>
#include "pointmatcher/PointMatcher.h"
#include <nlohmann/json.hpp>
#include <fstream>


using json = nlohmann::json;
typedef pcl::PointXYZINormal PointType;
typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

extern std::string config_path;
extern json config;
extern std::string OUTPUT_DIR;


void loadConfig();
template <typename CloudType>
void farthestPointSampling(
    const CloudType& source_cloud,
    int num_samples,
    std::vector<int>& sampled_indices);
// void farthestPointSampling(
//     const pcl::PointCloud<PointType>::Ptr& source_cloud,
//     int num_samples,
//     std::vector<int>& sampled_indices);

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}

inline double pointDistance(const PointType& p1, const PointType& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2) + std::pow(p1.z - p2.z, 2));
}

inline void getXYZ(
    const pcl::PointCloud<PointType>::Ptr& inputCloud,
    std::vector<Eigen::Vector3d>& outputCloud)
{
    outputCloud.clear();  // 清除输出容器中的旧数据
    outputCloud.reserve(inputCloud->size());  // 预分配内存，提升效率

    for (const auto& point : inputCloud->points) {
        Eigen::Vector3d xyzPoint(point.x, point.y, point.z);
        outputCloud.push_back(xyzPoint);  // 将Eigen::Vector3d添加到输出容器中
    }
}

inline void getNormals(
    const pcl::PointCloud<PointType>::Ptr& inputCloud,
    std::vector<Eigen::Vector3d>& outputNormals)
{
    outputNormals.clear();
    outputNormals.reserve(inputCloud->points.size());

    for (const auto& point : inputCloud->points) {
        Eigen::Vector3d normalVector(point.normal_x, point.normal_y, point.normal_z);
        outputNormals.push_back(normalVector);
    }
}

inline pcl::PointCloud<PointType>::Ptr convertDataPointsToPointCloud(const DP& dp_cloud)
{
    pcl::PointCloud<PointType>::Ptr pcl_cloud(new pcl::PointCloud<PointType>);

    for (int i = 0; i < dp_cloud.features.cols(); ++i) {
        PointType point;
        point.x = dp_cloud.features(0, i);  // x 坐标
        point.y = dp_cloud.features(1, i);  // y 坐标
        point.z = dp_cloud.features(2, i);  // z 坐标
        point.intensity = 1.0;              // intensity 设置为 1

        /* ---------- ATTENTION: LIBPOINTMATCHER HAS BUGS, HERE NORMAL IS ACTUALLY TANGENT IN LIBPOINTMATCHER ----------*/
        Eigen::Vector3f normal = dp_cloud.descriptors.block<3,1>(6, i);
        point.normal_x = normal(0);  // normal_x
        point.normal_y = normal(1);  // normal_y
        point.normal_z = normal(2);  // normal_z

        // 使用 surfaceness 作为 curvature
        point.curvature = dp_cloud.descriptors(0, i);

        pcl_cloud->push_back(point);
    }

    return pcl_cloud;
}

template<typename PointT>
inline void computeBoundingBox(const pcl::PointCloud<PointT>& cloud, PointT& min_pt, PointT& max_pt) {
    if (cloud.empty()) {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<double>::max();
    max_pt.x = max_pt.y = max_pt.z = -std::numeric_limits<double>::max();

    for (const auto& point : cloud.points) {
        if (point.x < min_pt.x) min_pt.x = point.x;
        if (point.y < min_pt.y) min_pt.y = point.y;
        if (point.z < min_pt.z) min_pt.z = point.z;

        if (point.x > max_pt.x) max_pt.x = point.x;
        if (point.y > max_pt.y) max_pt.y = point.y;
        if (point.z > max_pt.z) max_pt.z = point.z;
    }
}

#endif // COMMON_H