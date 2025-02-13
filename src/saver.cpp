#include <fstream>
#include <iomanip>
#include <string>
#include <cstring>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pointmatcher/PointMatcher.h>

#include "saver.h"
#include <cassert>


using namespace PointMatcherSupport;


void savePointCloudToTxt(const typename pcl::PointCloud<PointType>::Ptr& cloud, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Couldn't open file for writing!" << std::endl;
        return;
    }

    for (const auto& point : cloud->points) {
        // 对于 PointXYZINormal 类型的点，手动提取和写入每个字段
        file << point.x << " " << point.y << " " << point.z << " " 
             << point.intensity << " " 
             << point.normal_x << " " << point.normal_y << " " << point.normal_z << " " 
             << point.curvature << "\n";
    }

    file.close();
}

void savePoseToFile(const Eigen::Matrix4d& pose, const std::string& filename, const std::string& timestamp) {
    std::ofstream file(filename, std::ios::app);
    Eigen::Quaterniond q(pose.block<3, 3>(0, 0));
    file << std::fixed << std::setprecision(6)
         << timestamp << " "
         << pose(0, 3) << " " << pose(1, 3) << " " << pose(2, 3) << " "
         << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    file.close();
}

void saveMarkerToFile(const visualization_msgs::Marker& marker, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Couldn't open file for writing: " << filename << std::endl;
        return;
    }

    // 写入顶点数据
    for (size_t i = 0; i < marker.points.size(); i++) {
        const geometry_msgs::Point& p = marker.points[i];
        file << "v " << p.x << " " << p.y << " " << p.z << std::endl;
    }

    // 写入线段数据
    for (size_t i = 0; i < marker.points.size(); i += 2) {
        file << "l " << i + 1 << " " << i + 2 << std::endl;  // OBJ 的索引从 1 开始
    }

    file.close();
    std::cout << "PCA Marker saved to " << filename << std::endl;
}

void saveThresholdFile(const std::string& filename, double number) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return;
    }
    file << std::fixed << std::setprecision(6) << number << std::endl;
    file.close();
}

void saveCloudFeaturesAndDescriptors(const DP& dp_cloud, const std::string& filename) {
    std::ofstream output_file(filename);
    if (!output_file.is_open()) {
        std::cerr << "无法打开输出文件: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < dp_cloud.features.cols(); ++i) {
        // Output x, y, z coordinates (features)
        output_file << dp_cloud.features(0, i) << " "    // x
                    << dp_cloud.features(1, i) << " "    // y
                    << dp_cloud.features(2, i) << " ";   // z

        // Output all 22 descriptor values for the current point
        for (int j = 0; j < dp_cloud.descriptors.rows(); ++j) {
            output_file << dp_cloud.descriptors(j, i);
            if (j < dp_cloud.descriptors.rows() - 1)
                output_file << " ";  // Add a space between descriptor values
        }
        output_file << "\n";  // New line for the next point
    }

    output_file.close();
}

void saveMatchedPointsToFile(
    const std::vector<Eigen::Vector3d>& source_cloud,
    const std::vector<Eigen::Vector3d>& matched_cloud,
    const std::string& filename)
{
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Couldn't open file for writing matched points!" << std::endl;
        return;
    }

    for (size_t i = 0; i < source_cloud.size(); ++i) {
        const Eigen::Vector3d& source_pt = source_cloud[i];
        const Eigen::Vector3d& matched_pt = matched_cloud[i];
        
        file << source_pt(0) << " " << source_pt(1) << " " << source_pt(2) << " "
             << matched_pt(0) << " " << matched_pt(1) << " " << matched_pt(2) << "\n";
    }

    file.close();
}

sensor_msgs::PointCloud2 libPointMatcherToRosMsg(const DP& dpCloud, 
                                                 const std::string& frame_id, const ros::Time& stamp)
{
    sensor_msgs::PointCloud2 rosCloud;
    typedef sensor_msgs::PointField PF;

    // Setup basic metadata
    rosCloud.header.frame_id = frame_id;
    rosCloud.header.stamp = stamp;
    rosCloud.height = 1;
    rosCloud.width = dpCloud.features.cols();
    rosCloud.is_bigendian = false;
    rosCloud.is_dense = true;

    size_t offset = 0;
    uint8_t dataType = PF::FLOAT32;
    size_t scalarSize = 4;

    // Add feature fields (x, y, z, pad)
    for (const auto& label : dpCloud.featureLabels)
    {
        if (label.text == "pad") continue;  // Skip padding
        PF pointField;
        pointField.name = label.text;
        pointField.offset = offset;
        pointField.datatype = dataType;
        pointField.count = label.span;
        rosCloud.fields.push_back(pointField);
        offset += label.span * scalarSize;
    }

    // Add descriptor fields (surfaceness, curveness, etc.)
    for (const auto& label : dpCloud.descriptorLabels)
    {
        PF pointField;
        pointField.name = label.text;
        pointField.offset = offset;
        pointField.datatype = dataType;
        pointField.count = label.span;
        rosCloud.fields.push_back(pointField);
        offset += label.span * scalarSize;
    }

    // Add time fields (if available)
    if (dpCloud.times.rows() > 0)
    {
        PF pointField;
        pointField.name = "time";
        pointField.offset = offset;
        pointField.datatype = dataType;
        pointField.count = dpCloud.times.rows();
        rosCloud.fields.push_back(pointField);
        offset += scalarSize * dpCloud.times.rows();
    }

    // Allocate data buffer
    rosCloud.point_step = offset;
    rosCloud.row_step = rosCloud.point_step * rosCloud.width;
    rosCloud.data.resize(rosCloud.row_step * rosCloud.height);

    // Fill the cloud with feature and descriptor data
    for (size_t pt = 0; pt < rosCloud.width; ++pt)
    {
        uint8_t* ptr = &rosCloud.data[pt * rosCloud.point_step];

        // Copy features (x, y, z)
        memcpy(ptr, dpCloud.features.block<3,1>(0, pt).data(), 3 * scalarSize);
        ptr += 3 * scalarSize;

        // Copy descriptors (surfaceness, curveness, etc.)
        size_t descriptor_offset = 0;
        for (const auto& label : dpCloud.descriptorLabels)
        {
            size_t label_size = label.span * scalarSize;
            memcpy(ptr, &dpCloud.descriptors(descriptor_offset, pt), label_size);
            ptr += label_size;
            descriptor_offset += label.span;  // 调整 offset 到下一个 descriptor
        }

        // Copy time (if available)
        if (dpCloud.times.rows() > 0)
        {
            memcpy(ptr, &dpCloud.times(0, pt), dpCloud.times.rows() * scalarSize);
        }
    }

    return rosCloud;
}

DP rosMsgToLibPointMatcherCloud(const sensor_msgs::PointCloud2& rosMsg)
{
    typedef typename DP::Label Label;
    typedef typename DP::Labels Labels;
    typedef typename DP::View View;

    // Check if fields are empty
    if (rosMsg.fields.empty())
        return DP();

    // Labels for features (x, y, z) and descriptors (like normals)
    Labels featLabels;
    Labels descLabels;
    Labels timeLabels;
    
    // Add feature labels
    featLabels.push_back(Label("x", 1));
    featLabels.push_back(Label("y", 1));
    featLabels.push_back(Label("z", 1));
    featLabels.push_back(Label("pad", 1));  // padding label

    // Add descriptors based on your custom labels
    descLabels.push_back(Label("surfaceness", 1));
    descLabels.push_back(Label("curveness", 1));
    descLabels.push_back(Label("pointness", 1));
    descLabels.push_back(Label("normals", 3));
    descLabels.push_back(Label("tangents", 3));
    descLabels.push_back(Label("labels", 1));
    descLabels.push_back(Label("sticks", 4));
    descLabels.push_back(Label("plates", 7));
    descLabels.push_back(Label("balls", 1));

    // Time labels if necessary
    timeLabels.push_back(Label("time", 1));

    // Create cloud with labels
    const unsigned pointCount = rosMsg.width * rosMsg.height;
    DP cloud(featLabels, descLabels, timeLabels, pointCount);

    // Fill feature data (x, y, z)
    View xView = cloud.getFeatureViewByName("x");
    View yView = cloud.getFeatureViewByName("y");
    View zView = cloud.getFeatureViewByName("z");
    View padView = cloud.getFeatureViewByName("pad");
    padView.setConstant(1);  // set padding to 1

    // Fill descriptor data
    View surfacenessView = cloud.getDescriptorViewByName("surfaceness");
    View curvenessView = cloud.getDescriptorViewByName("curveness");
    View pointnessView = cloud.getDescriptorViewByName("pointness");
    View normalsView = cloud.getDescriptorViewByName("normals");
    View tangentsView = cloud.getDescriptorViewByName("tangents");
    View labelsView = cloud.getDescriptorViewByName("labels");
    View sticksView = cloud.getDescriptorViewByName("sticks");
    View platesView = cloud.getDescriptorViewByName("plates");
    View ballsView = cloud.getDescriptorViewByName("balls");

    // Read each point and fill the data
    const uint8_t* dataPtr = rosMsg.data.data();
    const uint32_t pointStep = rosMsg.point_step;
    for (unsigned int pt = 0; pt < pointCount; ++pt)
    {
        const uint8_t* pointData = dataPtr + pt * pointStep;

        // Read x, y, z coordinates (assuming 32-bit float)
        std::memcpy(&xView(0, pt), pointData + rosMsg.fields[0].offset, sizeof(float));
        std::memcpy(&yView(0, pt), pointData + rosMsg.fields[1].offset, sizeof(float));
        std::memcpy(&zView(0, pt), pointData + rosMsg.fields[2].offset, sizeof(float));

        // Read custom descriptors
        std::memcpy(&surfacenessView(0, pt), pointData + rosMsg.fields[3].offset, sizeof(float));
        std::memcpy(&curvenessView(0, pt), pointData + rosMsg.fields[4].offset, sizeof(float));
        std::memcpy(&pointnessView(0, pt), pointData + rosMsg.fields[5].offset, sizeof(float));
        std::memcpy(&normalsView(0, pt), pointData + rosMsg.fields[6].offset, sizeof(float) * 3);
        std::memcpy(&tangentsView(0, pt), pointData + rosMsg.fields[7].offset, sizeof(float) * 3);
        std::memcpy(&labelsView(0, pt), pointData + rosMsg.fields[8].offset, sizeof(float));
        std::memcpy(&sticksView(0, pt), pointData + rosMsg.fields[9].offset, sizeof(float) * 4);
        std::memcpy(&platesView(0, pt), pointData + rosMsg.fields[10].offset, sizeof(float) * 7);
        std::memcpy(&ballsView(0, pt), pointData + rosMsg.fields[11].offset, sizeof(float));
    }

    return cloud;
}

void publishPointCloud(
    const pcl::PointCloud<PointType>::Ptr& cloud,
    ros::Publisher& publisher,
    const ros::Time& stamp,
    const std::string& frame_id)
{
    sensor_msgs::PointCloud2 cloudMsg;
    pcl::toROSMsg(*cloud, cloudMsg);
    cloudMsg.header.stamp = stamp;
    cloudMsg.header.frame_id = frame_id;
    publisher.publish(cloudMsg);
}

void publishDataPointsAsPointCloud2(const DP& dp_cloud, const std::string& frame_id, const ros::Publisher& publisher, const ros::Time& stamp) {
    // 使用 pointmatcher_ros 将 DataPoints 转换为 sensor_msgs::PointCloud2
    sensor_msgs::PointCloud2 ros_cloud = libPointMatcherToRosMsg(dp_cloud, frame_id, stamp);
    publisher.publish(ros_cloud);  // 发布消息
}

void pubPath(const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation,
    nav_msgs::Path& path,
    ros::Publisher& mcu_path_pub_,
    const ros::Time& stamp,
    const std::string& frame_id)
    {
        geometry_msgs::PoseStamped this_pose_stamped;

        // 设置位置
        this_pose_stamped.pose.position.x = position(0);
        this_pose_stamped.pose.position.y = position(1);
        this_pose_stamped.pose.position.z = position(2);

        // 设置四元数
        this_pose_stamped.pose.orientation.x = orientation.x();
        this_pose_stamped.pose.orientation.y = orientation.y();
        this_pose_stamped.pose.orientation.z = orientation.z();
        this_pose_stamped.pose.orientation.w = orientation.w();

        // 设置时间戳和坐标系
        this_pose_stamped.header.stamp = stamp;
        this_pose_stamped.header.frame_id = frame_id;

        // 将该位姿添加到路径中
        path.poses.push_back(this_pose_stamped);

        // 发布路径消息
        mcu_path_pub_.publish(path);
}

void visualizePCAFeatures(
    const pcl::PointCloud<PointType>::Ptr& laserCloud,
    visualization_msgs::Marker& pcaMarker)
{
    // 清空 marker 中的点和颜色信息
    pcaMarker.points.clear();
    pcaMarker.colors.clear();

    for (const auto& point : laserCloud->points)
    {
        // 获取原始点的位置
        geometry_msgs::Point p;
        p.x = point.x;
        p.y = point.y;
        p.z = point.z;

        // 获取法向量并将其标准化
        Eigen::Vector3d normal(point.normal_x, point.normal_y, point.normal_z);
        Eigen::Vector3d normalized_normal = normal.normalized();

        // 设置法向量的可视化终点
        geometry_msgs::Point p1;
        p1.x = p.x + normalized_normal(0);
        p1.y = p.y + normalized_normal(1);
        p1.z = p.z + normalized_normal(2);

        // 将原始点和法向量的终点添加到 marker 中
        pcaMarker.points.push_back(p);
        pcaMarker.points.push_back(p1);

        // 设置颜色（这里默认红色表示法向量）
        std_msgs::ColorRGBA color;
        color.r = 1.0;
        color.g = 0.0;
        color.b = 0.0;
        color.a = 1.0;
        pcaMarker.colors.push_back(color);
        pcaMarker.colors.push_back(color);
    }
}
