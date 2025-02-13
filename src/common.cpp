#include "common.h"

std::string config_path = "/home/gaoyang/catkin_ws/src/planetary_slam/config.json";
json config;
std::string OUTPUT_DIR;


void loadConfig() {
    std::ifstream config_file(config_path);
    if (config_file.is_open()) {
        config = json::parse(config_file);
        OUTPUT_DIR = config["saver"]["output_dir"];
        config_file.close();
    } else {
        throw std::runtime_error("无法打开配置文件：" + config_path);
    }
}

template <typename CloudType>
void farthestPointSampling(
    const CloudType& source_cloud,
    int num_samples,
    std::vector<int>& sampled_indices)
{
    // 初始化点云大小和获取点的方法
    int cloud_size;
    std::function<Eigen::Vector3d(int)> getPoint;

    // 根据类型选择获取点的方法
    if constexpr (std::is_same_v<CloudType, pcl::PointCloud<PointType>::Ptr>) {
        cloud_size = source_cloud->points.size();
        getPoint = [&](int index) -> Eigen::Vector3d {
            const auto& pt = source_cloud->points[index];
            return Eigen::Vector3d(pt.x, pt.y, pt.z);
        };
    } else if constexpr (std::is_same_v<CloudType, std::vector<Eigen::Vector3d>>) {
        cloud_size = source_cloud.size();
        getPoint = [&](int index) -> Eigen::Vector3d {
            return source_cloud[index];
        };
    } else {
        throw std::invalid_argument("Unsupported cloud type");
    }

    // 初始化最小距离缓存
    std::vector<double> min_distances(cloud_size, std::numeric_limits<double>::infinity());

    // 随机选择第一个点
    int first_index = rand() % cloud_size;
    sampled_indices.push_back(first_index);

    // 更新距离缓存
    Eigen::Vector3d first_point = getPoint(first_index);
    for (int i = 0; i < cloud_size; ++i) {
        min_distances[i] = (first_point - getPoint(i)).norm();
    }

    // 选择其余的点
    for (int sample_count = 1; sample_count < num_samples; ++sample_count) {
        double max_dist = -1.0;
        int farthest_index = -1;

        // 找到最远的点
        for (int i = 0; i < cloud_size; ++i) {
            if (std::find(sampled_indices.begin(), sampled_indices.end(), i) == sampled_indices.end()) {
                if (min_distances[i] > max_dist) {
                    max_dist = min_distances[i];
                    farthest_index = i;
                }
            }
        }

        // 添加到采样结果
        sampled_indices.push_back(farthest_index);

        // 更新最小距离缓存
        Eigen::Vector3d farthest_point = getPoint(farthest_index);
        for (int i = 0; i < cloud_size; ++i) {
            min_distances[i] = std::min(min_distances[i], (farthest_point - getPoint(i)).norm());
        }
    }
}

template void farthestPointSampling<std::vector<Eigen::Vector3d>>(const std::vector<Eigen::Vector3d>&, int, std::vector<int>&);
template void farthestPointSampling<boost::shared_ptr<pcl::PointCloud<PointType>>>(const boost::shared_ptr<pcl::PointCloud<PointType>>&, int, std::vector<int>&);

// void farthestPointSampling(
//     const pcl::PointCloud<PointType>::Ptr& source_cloud,
//     int num_samples,
//     std::vector<int>& sampled_indices)
// {
//     int cloud_size = source_cloud->points.size();
//     std::vector<double> min_distances(cloud_size, std::numeric_limits<double>::infinity()); // 存储每个点到已选点集的最小距离

//     // 随机选择第一个点
//     int first_index = rand() % cloud_size;
//     sampled_indices.push_back(first_index);

//     // 更新距离缓存
//     for (int i = 0; i < cloud_size; ++i) {
//         min_distances[i] = std::min(min_distances[i], pointDistance(source_cloud->points[first_index], source_cloud->points[i]));
//     }

//     // 选择其余的点
//     for (int sample_count = 1; sample_count < num_samples; ++sample_count) {
//         // 选择距离当前已选点集最远的点
//         double max_dist = -1.0;
//         int farthest_index = -1;
//         for (int i = 0; i < cloud_size; ++i) {
//             if (std::find(sampled_indices.begin(), sampled_indices.end(), i) == sampled_indices.end()) {
//                 if (min_distances[i] > max_dist) {
//                     max_dist = min_distances[i];
//                     farthest_index = i;
//                 }
//             }
//         }

//         // 添加到采样结果
//         sampled_indices.push_back(farthest_index);

//         // 更新距离缓存
//         for (int i = 0; i < cloud_size; ++i) {
//             min_distances[i] = std::min(min_distances[i], pointDistance(source_cloud->points[farthest_index], source_cloud->points[i]));
//         }
//     }
// }
