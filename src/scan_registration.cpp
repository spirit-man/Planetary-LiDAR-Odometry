#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>

#include "common.h"
#include "tic_toc.h"
#include "imls_icp.h"
#include "saver.h"
#include "range_image.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <opencv/cv.h>
#include <boost/functional/hash.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nabo/nabo.h>

#include <pointmatcher/PointMatcher.h>
#include "pointmatcher/DataPointsFilters/utils/sparsetv.h"
#include "pointmatcher/DataPointsFilters/Saliency.h"

#include "CSF.h"
#include "point_cloud.h" 

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>


using std::atan2;
using std::cos;
using std::sin;

const float scanPeriod = 0.1;
const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;
float AZIMUTH_RESOLUTION = 0.9;
bool PUB_EACH_LINE = false;
float MINIMUM_RANGE = 0.5; 
float MAXIMUM_RANGE = 120;
int frame = 1;
Eigen::Vector3f z_axis(0, 0, 1);

// filtered points
ros::Publisher pubLaserCloud;

ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

// pca visualization
ros::Publisher pubPCAFeatures;

// points in other regions: divided as flat features and others
ros::Publisher pubPointsFlat;
ros::Publisher pubPointsLessFlat;

// pubilsh points as datapoints
ros::Publisher pubLaserCloudDP;
ros::Publisher pubPointsFlatDP;

pcl::PointCloud<PointType>::Ptr last_pcl_cloud(new pcl::PointCloud<PointType>());


template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float min_thres, float max_thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < min_thres * min_thres
             || cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z > max_thres * max_thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

bool findNearestPoint(const PointType& query_point, pcl::KdTreeFLANN<PointType>::Ptr kdtree, 
                      std::string neighbor_scan, int& nearest_index, float knn_distance_threshold)
{
    if (neighbor_scan == "kdtree") {
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        if (kdtree->nearestKSearch(query_point, 1, indices, distances) > 0 && distances[0] < knn_distance_threshold) {
            nearest_index = indices[0];
            return true;
        }
    } 
    else if (neighbor_scan == "index") {
        return true;
    }
    else
    {
        std::cerr << "Invalid NEIGHBOR_SCAN for computing normal!" << std::endl;
    }
    return false;
}

bool checkPlaneValidity(const Eigen::MatrixXf& points, const Eigen::Vector3f& normal, float distance_threshold, float valid_points_threshold)
{
    int count = points.rows();
    int valid_points = 0;
    Eigen::Vector3f centroid = points.topRows(count).colwise().mean();

    // 计算平面方程: normal*(x - centroid) = 0
    for (int i = 0; i < count; i++) {
        // 计算每个点到平面的距离
        Eigen::Vector3f point(points.row(i));
        float distance = std::abs(normal.dot(point - centroid));

        if (distance < distance_threshold) {
            valid_points++;
        }
    }

    return valid_points >= valid_points_threshold * count;
}

void computeNormalPCA(const std::vector<pcl::PointCloud<PointType>>& laserCloudScans, int scanID, int idx, float& lambda1, float& lambda2, float& lambda3, Eigen::Matrix3f& eigen_vectors, const std::vector<pcl::KdTreeFLANN<PointType>::Ptr>& kdtrees,
    int window_size, int iter_step, float knn_distance_threshold, std::string neighbor_scan, float distance_threshold, float valid_points_threshold)
{
    int num = 3 * (int(2 * window_size / iter_step) + 1);
    Eigen::MatrixXf points(num, 3); // 使用动态大小矩阵
    int count = 0;

    // 当前扫描线
    for (int i = -window_size; i <= window_size; i += iter_step) {
        if (idx + i >= 0 && idx + i < laserCloudScans[scanID].points.size()) {
            points.row(count++) << laserCloudScans[scanID].points[idx + i].x, laserCloudScans[scanID].points[idx + i].y, laserCloudScans[scanID].points[idx + i].z;
        }
    }

    // 上一个扫描线
    if (scanID > 0) {
        int neighborIdx = idx;
        if (findNearestPoint(laserCloudScans[scanID].points[idx], kdtrees[scanID - 1], neighbor_scan, neighborIdx, knn_distance_threshold))
        {
            for (int i = -window_size; i <= window_size; i += iter_step) {
                if (neighborIdx + i >= 0 && neighborIdx + i < laserCloudScans[scanID - 1].points.size()) {
                    points.row(count++) << laserCloudScans[scanID - 1].points[neighborIdx + i].x, laserCloudScans[scanID - 1].points[neighborIdx + i].y, laserCloudScans[scanID - 1].points[neighborIdx + i].z;
                }
            }
        }
    }

    // 下一个扫描线
    if (scanID < laserCloudScans.size() - 1) {
        int neighborIdx = idx;
        if (findNearestPoint(laserCloudScans[scanID].points[idx], kdtrees[scanID + 1], neighbor_scan, neighborIdx, knn_distance_threshold))
        {
            for (int i = -window_size; i <= window_size; i += iter_step) {
                if (neighborIdx + i >= 0 && neighborIdx + i < laserCloudScans[scanID + 1].points.size()) {
                    points.row(count++) << laserCloudScans[scanID + 1].points[neighborIdx + i].x, laserCloudScans[scanID + 1].points[neighborIdx + i].y, laserCloudScans[scanID + 1].points[neighborIdx + i].z;
                }
            }
        }
    }

    if (count < num) {
        lambda1 = lambda2 = lambda3 = 0;
        return; // Not enough points for 3*scan
    }

    Eigen::Vector3f centroid = points.topRows(count).colwise().mean();
    Eigen::MatrixXf centered = points.topRows(count).rowwise() - centroid.transpose();
    Eigen::Matrix3f covariance = (centered.adjoint() * centered) / float(count - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Vector3f eigen_values = eigen_solver.eigenvalues();
    eigen_vectors = eigen_solver.eigenvectors();
    
    // 获取法向量(对应最小特征值的特征向量)
    Eigen::Vector3f normal = eigen_vectors.col(0);  // 最小特征值对应的特征向量
    
    // 检查是否有足够的点支持这个平面(至少1/3的点)
    if(!checkPlaneValidity(points, normal, distance_threshold, valid_points_threshold)) {
        // 如果没有足够的点支持,则将特征值设为-1并返回
        lambda1 = lambda2 = lambda3 = -1;
        return;
    }

    // Eigen 特征值默认是按升序排列的：lambda_min, lambda_mid, lambda_max
    // 降序排列 lambda1 > lambda2 > lambda3，并同步调整特征向量的顺序。
    lambda1 = eigen_values(2);  // 最大特征值
    lambda2 = eigen_values(1);  // 中间特征值
    lambda3 = eigen_values(0);  // 最小特征值

    // 调整特征向量的顺序，使之与特征值对应
    eigen_vectors.col(0).swap(eigen_vectors.col(2));  // 交换第一列和第三列
}

void computeNormalCrossProduct(const std::vector<pcl::PointCloud<PointType>>& laserCloudScans, int scanID, int idx, const std::vector<pcl::KdTreeFLANN<PointType>::Ptr>& kdtrees, float knn_distance_threshold, std::string neighbor_scan, Eigen::Vector3f& normal)
{
    Eigen::Vector3f forward, backward, up, down;
    if (idx - 1 >= 0 && idx + 1 < laserCloudScans[scanID].points.size())
    {
        forward << laserCloudScans[scanID].points[idx + 1].x, laserCloudScans[scanID].points[idx + 1].y, laserCloudScans[scanID].points[idx + 1].z;
        backward << laserCloudScans[scanID].points[idx - 1].x, laserCloudScans[scanID].points[idx - 1].y, laserCloudScans[scanID].points[idx - 1].z;
    }
    else
    {
        return;
    }

    // 上一个扫描线
    if (scanID > 0) {
        int neighborIdx = idx;
        if (findNearestPoint(laserCloudScans[scanID].points[idx], kdtrees[scanID - 1], neighbor_scan, neighborIdx, knn_distance_threshold))
        {
            if (neighborIdx >= 0 && neighborIdx < laserCloudScans[scanID - 1].points.size())
            {
                up << laserCloudScans[scanID - 1].points[neighborIdx].x, laserCloudScans[scanID - 1].points[neighborIdx].y, laserCloudScans[scanID - 1].points[neighborIdx].z;
            }
            else
            {
                return;
            }
        }
    }

    // 下一个扫描线
    if (scanID < laserCloudScans.size() - 1) {
        int neighborIdx = idx;
        if (findNearestPoint(laserCloudScans[scanID].points[idx], kdtrees[scanID + 1], neighbor_scan, neighborIdx, knn_distance_threshold))
        {
            if (neighborIdx >= 0 && neighborIdx < laserCloudScans[scanID + 1].points.size())
            {
                down << laserCloudScans[scanID + 1].points[neighborIdx].x, laserCloudScans[scanID + 1].points[neighborIdx].y, laserCloudScans[scanID + 1].points[neighborIdx].z;
            }
            else
            {
                return;
            }
        }
    }

    normal = ((forward - backward).cross(up - down)).normalized(); 
}

void computeGeometricFeatures(
    const Eigen::MatrixXf& eigenvalues_matrix, 
    Eigen::MatrixXf& features, 
    float planarity_threshold,
    std::vector<int>& candidate_indices) 
{
    int num_points = eigenvalues_matrix.cols();
    candidate_indices.clear();  // 清空候选点索引

    features.resize(num_points, 8);  // 每个点8个特征值，调整大小

    // 计算几何特征
    Eigen::ArrayXf lambda1 = eigenvalues_matrix.row(0).array();
    Eigen::ArrayXf lambda2 = eigenvalues_matrix.row(1).array();
    Eigen::ArrayXf lambda3 = eigenvalues_matrix.row(2).array();

    // 计算 Sum
    Eigen::ArrayXf sum = lambda1 + lambda2 + lambda3;
    // 计算 Omnivariance
    Eigen::ArrayXf omnivariance = (lambda1 * lambda2 * lambda3).pow(1.0f / 3.0f);
    // 计算 Eigenentropy
    Eigen::ArrayXf eigenentropy = -(lambda1 * lambda1.log() + lambda2 * lambda2.log() + lambda3 * lambda3.log());
    // 计算 Anisotropy
    Eigen::ArrayXf anisotropy = (lambda1 - lambda3) / lambda1;
    // 计算 Linearity
    Eigen::ArrayXf linearity = (lambda1 - lambda2) / lambda1;
    // 计算 Planarity
    Eigen::ArrayXf planarity = (lambda2 - lambda3) / lambda1;
    // 计算 Surface variation
    Eigen::ArrayXf surface_variation = lambda3 / (lambda1 + lambda2 + lambda3);
    // 计算 Sphericity
    Eigen::ArrayXf sphericity = lambda3 / lambda1;

    features.col(0) = sum;
    features.col(1) = omnivariance;
    features.col(2) = eigenentropy;
    features.col(3) = anisotropy;
    features.col(4) = linearity;
    features.col(5) = planarity;
    features.col(6) = surface_variation;
    features.col(7) = sphericity;

    // 判断平面值的大小，筛选符合条件的点
    for (int i = 0; i < num_points; ++i) {
        if (planarity[i] > planarity_threshold) {
            candidate_indices.push_back(i);  // 满足条件，加入候选点索引
        }
    }
}

bool compareXYZ(const PointType& a, const PointType& b) {
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}


// customize tensor voting, use pca features as encoding
template <typename T>
class CustomTensorVoting : public TensorVoting<T> {
public:
    // 继承构造函数
    using TensorVoting<T>::TensorVoting;

    // 重写 refine 函数，使用自定义逻辑
    void customRefineWithEigen(const typename PointMatcher<T>::DataPoints& pts,
                               const Eigen::MatrixXf& eigenvalues,
                               const Eigen::MatrixXf& eigenvectors) {
        myCustomFunctionWithEigen(pts, eigenvalues, eigenvectors);  // 使用传入的特征值和特征向量 encode with aware tensor
        // this->ballVote(pts, true);
        this->decompose();      // 调用继承的分解函数
        this->toDescriptors();  // 调用继承的描述符生成
    }

private:
    // 自定义的处理函数，使用预计算的 eigenvalues 和 eigenvectors
    void myCustomFunctionWithEigen(const typename PointMatcher<T>::DataPoints& pts,
                                   const Eigen::MatrixXf& eigenvalues,
                                   const Eigen::MatrixXf& eigenvectors) {
        const std::size_t nbPts = pts.getNbPoints();
        this->sparseStick.resize(nbPts);
	    this->sparsePlate.resize(nbPts);
	    this->sparseBall.resize(nbPts);
        this->tensors.resize(nbPts, 1);

        for (std::size_t i = 0; i < nbPts; ++i) {
            // 获取传入的特征值和特征向量
            Eigen::Vector3f eigenVa = eigenvalues.col(i).array().abs();
            Eigen::VectorXf eigenVe = eigenvectors.col(i);

            const T lambda1 = eigenVa.maxCoeff();   // 最大特征值
            const T lambda3 = eigenVa.minCoeff();   // 最小特征值
            const T lambda2 = eigenVa.sum() - (lambda1 + lambda3);  // 中间特征值

            if (lambda1 >= lambda2 && lambda2 >= lambda3) {
                // decompose + encode with aware tensor
                Eigen::Vector3f stickTail = eigenVe.segment(0, 3);
                const typename TensorVoting<T>::Tensor S = stickTail * stickTail.transpose();

                Eigen::Vector3f plateTail = eigenVe.segment(3, 3);
                const typename TensorVoting<T>::Tensor P = stickTail * stickTail.transpose() + plateTail * plateTail.transpose();

                // ommit norm, norm = 1
                this->tensors(i) = ((lambda1 - lambda2) / this->k) * S + (lambda3 / this->k) * P;
       
            }
            else
            {
                // encode with unit ball
                this->tensors(i) = Eigen::Matrix<T, 3, 3>::Identity();
            }
        }
    }
};

template <typename T>
class CustomSaliencyDataPointsFilter : public SaliencyDataPointsFilter<T> {
public:
    // 使用父类的构造函数
    using SaliencyDataPointsFilter<T>::SaliencyDataPointsFilter;

    // 重写 inPlaceFilter 方法，增加特征值和特征向量的输入
    void customInPlaceFilter(typename PointMatcher<T>::DataPoints& cloud,
                             const Eigen::MatrixXf& eigenvalues,
                             const Eigen::MatrixXf& eigenvectors) {
        const std::size_t nbPts = cloud.getNbPoints();
        
        // 使用自定义的 TensorVoting 类
        CustomTensorVoting<T> tv(this->sigma, this->k);

        // 调用自定义的 refine 函数，传入特征值和特征向量
        tv.customRefineWithEigen(cloud, eigenvalues, eigenvectors);

        // 继续执行后续步骤，例如计算显著性等
        tv.disableBallComponent();
        tv.cfvote(cloud, true);
        tv.decompose();
        tv.toDescriptors();

        Eigen::MatrixXf labels = Eigen::MatrixXf::Zero(1, nbPts);
        for (std::size_t i = 0; i < nbPts; ++i) {
            const T lambda1 = tv.surfaceness(i);
            const T lambda2 = tv.curveness(i);
            const T lambda3 = tv.pointness(i);

            int index;
            Eigen::VectorXf coeff(3);
            coeff << lambda3, (lambda2 - lambda3), (lambda1 - lambda2);
            coeff.maxCoeff(&index);

            labels(i) = index + 1;
        }

        // 添加描述符到点云
        try {
            cloud.addDescriptor("surfaceness", tv.surfaceness);
            cloud.addDescriptor("curveness", tv.curveness);
            cloud.addDescriptor("pointness", tv.pointness);

            if (this->keepNormals) {
                cloud.addDescriptor("normals", tv.normals);
                cloud.addDescriptor("tangents", tv.tangents);
            }
            if (this->keepLabels) {
                cloud.addDescriptor("labels", labels);
            }
            if (this->keepTensors) {
                cloud.addDescriptor("sticks", tv.sticks);
                cloud.addDescriptor("plates", tv.plates);
                cloud.addDescriptor("balls", tv.balls);
            }
        } catch (...) {
            std::cerr << "CustomSaliencyDataPointsFilter<T>::inPlaceFilter: Cannot add descriptors to pointcloud" << std::endl;
        }
    }
};


DP applySaliencyFilter(const pcl::PointCloud<PointType>::Ptr& cloud, const Eigen::MatrixXf& eigenvalues, const Eigen::MatrixXf& eigenvectors, size_t k = 50, float sigma = 0.2, bool keepNormals = true, bool keepLabels = true, bool keepTensors = true) {
    // 特征标签，包含 x, y, z 三个坐标
    DP::Labels featureLabels;
    featureLabels.push_back(DP::Label("x", 1));
    featureLabels.push_back(DP::Label("y", 1));
    featureLabels.push_back(DP::Label("z", 1));

    // 创建特征矩阵，行数为 4（x, y, z, homogeneous），列数为点云的大小
    Eigen::MatrixXf features(4, cloud->size());

    for (size_t i = 0; i < cloud->size(); ++i) {
        features(0, i) = cloud->points[i].x;
        features(1, i) = cloud->points[i].y;
        features(2, i) = cloud->points[i].z;
        features(3, i) = 1.0; // homogeneous 坐标
    }

    // 创建 DataPoints 对象
    DP dp_cloud(features, featureLabels);

    // 自定义 Saliency Filter 参数设置
    PM::Parameters params;
    params["k"] = std::to_string(k);                  // 邻居数
    params["sigma"] = std::to_string(sigma);          // Tensor Voting sigma
    params["keepNormals"] = keepNormals ? "1" : "0";  // 是否保留法线
    params["keepLabels"] = keepLabels ? "1" : "0";    // 是否保留标签
    params["keepTensors"] = keepTensors ? "1" : "0";  // 是否保留张量

    // 创建自定义的 SaliencyDataPointsFilter 对象
    CustomSaliencyDataPointsFilter<float> customFilter(params);

    // 调用自定义的 inPlaceFilter 函数，传入特征值和特征向量
    customFilter.customInPlaceFilter(dp_cloud, eigenvalues, eigenvectors);

    return dp_cloud;
}

void threeAxisSampling(
    const pcl::PointCloud<PointType>::Ptr& pcl_cloud,
    const Eigen::MatrixXf& eigenvalues_matrix,
    const std::vector<int>& candidate_indices,
    int points_per_list,
    std::vector<int>& sampled_indices)
{
    std::vector<std::pair<float, int>> lists[9];
    for (int idx : candidate_indices) {
        const auto& point = pcl_cloud->points[idx];
        Eigen::Vector3f p(point.x, point.y, point.z);
        Eigen::Vector3f n(point.normal_x, point.normal_y, point.normal_z);

        // 计算 a2D
        float aD = (sqrt(eigenvalues_matrix(1, idx)) - sqrt(eigenvalues_matrix(2, idx))) / sqrt(eigenvalues_matrix(0, idx));
        float a2D = aD * aD;

        // 计算9个值
        Eigen::Vector3f cross = p.cross(n);
        lists[0].emplace_back(a2D * cross.dot(Eigen::Vector3f(1, 0, 0)), idx);
        lists[1].emplace_back(-a2D * cross.dot(Eigen::Vector3f(1, 0, 0)), idx);
        lists[2].emplace_back(a2D * cross.dot(Eigen::Vector3f(0, 1, 0)), idx);
        lists[3].emplace_back(-a2D * cross.dot(Eigen::Vector3f(0, 1, 0)), idx);
        lists[4].emplace_back(a2D * cross.dot(Eigen::Vector3f(0, 0, 1)), idx);
        lists[5].emplace_back(-a2D * cross.dot(Eigen::Vector3f(0, 0, 1)), idx);
        lists[6].emplace_back(a2D * std::abs(n.dot(Eigen::Vector3f(1, 0, 0))), idx);
        lists[7].emplace_back(a2D * std::abs(n.dot(Eigen::Vector3f(0, 1, 0))), idx);
        lists[8].emplace_back(a2D * std::abs(n.dot(Eigen::Vector3f(0, 0, 1))), idx);
    }

    // 对每个list进行排序并采样
    // 多个list中有同样的点则多次采样
    for (auto& list : lists) {
        std::sort(list.begin(), list.end(), std::greater<>()); // 降序排列
        int count = 0;
        for (const auto& [value, idx] : list) {
            if (count >= points_per_list) break;
            sampled_indices.push_back(idx);
            count++;
        }
    }
}

// 计算球面直方图
std::vector<std::vector<std::vector<int>>> computeSphericalHistogram(
    const pcl::PointCloud<PointType>::Ptr& pcl_cloud,
    const std::vector<int>& candidate_indices,
    int azimuth_bins,
    int elevation_bins)
{
    std::vector<std::vector<std::vector<int>>> histogram(
        azimuth_bins,
        std::vector<std::vector<int>>(elevation_bins)
    );

    for (int idx : candidate_indices) {
        const PointType& pt = pcl_cloud->points[idx];
        Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);

        float azimuth = atan2(normal[1], normal[0]);
        float elevation = asin(normal[2]);

        if (azimuth < 0) azimuth += 2 * M_PI;
        elevation += M_PI / 2;

        int azimuth_idx = std::min(static_cast<int>(azimuth / (2 * M_PI / azimuth_bins)), azimuth_bins - 1);
        int elevation_idx = std::min(static_cast<int>(elevation / (M_PI / elevation_bins)), elevation_bins - 1);

        histogram[azimuth_idx][elevation_idx].push_back(idx);
    }

    return histogram;
}

void randomSampling(
    const std::vector<int>& candidate_indices,
    int max_points,
    std::vector<int>& sampled_indices)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> shuffled_indices = candidate_indices;
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), gen);

    int surface_count = std::min(max_points, static_cast<int>(shuffled_indices.size()));

    for (int i = 0; i < surface_count; ++i) {
        int idx = shuffled_indices[i];
        sampled_indices.push_back(idx);
    }
}

void normalSampling(
    const pcl::PointCloud<PointType>::Ptr& pcl_cloud,
    const std::vector<int>& candidate_indices,
    int azimuth_bins,
    int elevation_bins,
    int min_points_per_bin,
    int max_points_per_bin,
    std::string sampling_strategy,
    std::vector<int>& sampled_indices)
{
    auto histogram = computeSphericalHistogram(pcl_cloud, candidate_indices, azimuth_bins, elevation_bins);

    for (int az = 0; az < azimuth_bins; ++az) {
        for (int el = 0; el < elevation_bins; ++el) {
            auto& bin_points = histogram[az][el];
            int bin_size = bin_points.size();
            
            if (bin_size < min_points_per_bin) continue;
            
            if (bin_size > max_points_per_bin) {
                std::vector<int> sampled_indices_in_bin;
                if (sampling_strategy == "FPS"){
                    pcl::PointCloud<PointType>::Ptr sub_cloud(new pcl::PointCloud<PointType>);
                    for (int bin_idx : bin_points) {
                        sub_cloud->points.push_back(pcl_cloud->points[bin_idx]);
                    }
                    farthestPointSampling(sub_cloud, max_points_per_bin, sampled_indices_in_bin);
                    for (int bin_idx : sampled_indices_in_bin) 
                    {
                        sampled_indices.push_back(bin_points[bin_idx]);
                    }
                }
                else if (sampling_strategy == "random"){
                    randomSampling(bin_points, max_points_per_bin, sampled_indices_in_bin);
                    sampled_indices.insert(sampled_indices.end(), sampled_indices_in_bin.begin(), sampled_indices_in_bin.end());
                }
                else{
                    std::cerr << "Invalid SAMPLE_STRATEGY for sample method normal!" << std::endl;
                }
            } 
            else {
                sampled_indices.insert(sampled_indices.end(), bin_points.begin(), bin_points.end());
            }
        }
    }
}

void majorAxisSampling(
    const pcl::PointCloud<PointType>::Ptr& pcl_cloud,
    const pcl::PointCloud<PointType>::Ptr& last_pcl_cloud,
    const std::vector<int>& candidate_indices,
    float r,
    float r_proj,
    int max_total_points,
    int azimuth_bins,
    int elevation_bins,
    int min_points_per_bin,
    int max_points_per_bin,
    std::string sampling_strategy,
    std::vector<int>& sampled_indices)
{
    auto histogram = computeSphericalHistogram(pcl_cloud, candidate_indices, azimuth_bins, elevation_bins);

    // 计算每个bin的权重并采样
    std::vector<float> bin_weights(azimuth_bins * elevation_bins, 0.0f);
    int total_sampled_points = 0;
    
    for (int az = 0; az < azimuth_bins; ++az) {
        for (int el = 0; el < elevation_bins; ++el) {
            auto& bin_points = histogram[az][el];
            int bin_size = bin_points.size();

            if (bin_size < min_points_per_bin) continue;

            std::vector<int> sampled_indices_in_bin;
            if (bin_size > max_points_per_bin) {
                randomSampling(bin_points, max_points_per_bin, sampled_indices_in_bin);
            }
            else{
                sampled_indices_in_bin = bin_points;
            }

            std::vector<float> distances(sampled_indices_in_bin.size(), 0.0f);
            int valid_samples = 0;

            // 对bin中的点进行采样
            for (int idx : sampled_indices_in_bin) {
                Eigen::Vector3f pt(pcl_cloud->points[idx].x,
                                   pcl_cloud->points[idx].y,
                                   pcl_cloud->points[idx].z);
                Eigen::Vector3f normal(pcl_cloud->points[idx].normal);

                // 计算当前点到上一帧点云中与该点相近的点的平均距离
                std::vector<int> nearby_points;
                float avg_distance = 0.0f;
                for (int j = 0; j < last_pcl_cloud->points.size(); ++j) {
                    Eigen::Vector3f last_pt(last_pcl_cloud->points[j].x,
                                             last_pcl_cloud->points[j].y,
                                             last_pcl_cloud->points[j].z);
                    if ((pt - last_pt).norm() < r_proj && (pt - last_pt).cross(normal).norm() < r) {
                        nearby_points.push_back(j);
                    }
                }

                // 判断是否满足至少3个近邻点
                if (nearby_points.size() >= 3) {
                    // 计算平均距离
                    for (int k : nearby_points) {
                        Eigen::Vector3f last_pt(last_pcl_cloud->points[k].x,
                                                 last_pcl_cloud->points[k].y,
                                                 last_pcl_cloud->points[k].z);
                        avg_distance += (pt - last_pt).norm();
                    }
                    avg_distance /= nearby_points.size();

                    distances[valid_samples] = avg_distance;
                    valid_samples++;
                }
            }

            if (valid_samples >= 3) {
                // 计算平均距离作为权重
                float total_distance = 0.0f;
                for (float d : distances) {
                    total_distance += d;
                }
                bin_weights[az * elevation_bins + el] = total_distance / valid_samples;
            }
        }
    }

    // 计算归一化的权重
    float total_weight = 0.0f;
    for (float weight : bin_weights) {
        total_weight += weight;
    }

    for (float& weight : bin_weights) {
        weight /= total_weight;
    }

    // 根据权重进行采样
    for (int az = 0; az < azimuth_bins; ++az) {
        for (int el = 0; el < elevation_bins; ++el) {
            auto& bin_points = histogram[az][el];
            int bin_size = bin_points.size();
            if (bin_size < min_points_per_bin) continue;

            int points_to_sample = std::min(static_cast<int>(bin_weights[az * elevation_bins + el] * max_total_points), bin_size);
            
            std::vector<int> sampled_indices_in_bin;
            if (bin_size > points_to_sample) {
                if (sampling_strategy == "FPS") {
                    pcl::PointCloud<PointType>::Ptr sub_cloud(new pcl::PointCloud<PointType>);
                    for (int bin_idx : bin_points) {
                        sub_cloud->points.push_back(pcl_cloud->points[bin_idx]);
                    }
                    farthestPointSampling(sub_cloud, points_to_sample, sampled_indices_in_bin);
                    for (int bin_idx : sampled_indices_in_bin) {
                        sampled_indices.push_back(bin_points[bin_idx]);
                    }
                }
                else if (sampling_strategy == "random") {
                    randomSampling(bin_points, points_to_sample, sampled_indices_in_bin);
                    sampled_indices.insert(sampled_indices.end(), sampled_indices_in_bin.begin(), sampled_indices_in_bin.end());
                }
                else {
                    std::cerr << "Invalid SAMPLE_STRATEGY for sample method major_axis!" << std::endl;
                }
            } 
            else {
                sampled_indices.insert(sampled_indices.end(), bin_points.begin(), bin_points.end());
            }
        }
    }
}

void samplePointCloud(
    const std::string& sample_method,
    const pcl::PointCloud<PointType>::Ptr& pcl_cloud,
    const Eigen::MatrixXf& eigenvalues_matrix,
    const std::vector<int>& candidate_indices,
    int frame,
    std::vector<int>& sampled_indices)
{
    if (sample_method == "three_axis") {
        if (eigenvalues_matrix.rows() > 0 && eigenvalues_matrix.cols() > 0) {
            int points_per_list = config["scan_registration"]["sample_method"]["three_axis"]["points_per_list"];
            threeAxisSampling(pcl_cloud, eigenvalues_matrix, candidate_indices, points_per_list, sampled_indices);
        }
        else
        {
            std::cerr << "Eigenvalues matrix is not initialized! Choose pca to compute normal." << std::endl;
        }
    }
    else if (sample_method == "random") {
        int max_surface_points = config["scan_registration"]["sample_method"]["random"]["max_points"];  // 设置最大提取点数
        randomSampling(candidate_indices, max_surface_points, sampled_indices);
    }
    else if (sample_method == "normal" || (sample_method == "major_axis" && frame == 1)) {
        int azimuth_bins = config["scan_registration"]["sample_method"]["normal"]["azimuth_bins"];
        int elevation_bins = config["scan_registration"]["sample_method"]["normal"]["elevation_bins"];
        int min_points_per_bin = config["scan_registration"]["sample_method"]["normal"]["min_points_per_bin"];
        int max_points_per_bin = config["scan_registration"]["sample_method"]["normal"]["max_points_per_bin"];
        std::string sampling_strategy = config["scan_registration"]["sample_method"]["normal"]["sampling_strategy"];
        normalSampling(pcl_cloud, candidate_indices, azimuth_bins, elevation_bins, min_points_per_bin, max_points_per_bin, sampling_strategy, sampled_indices);
    }
    else if (sample_method == "major_axis" && frame != 1) {
        float r = config["scan_registration"]["sample_method"]["major_axis"]["r"];
        float r_proj = config["scan_registration"]["sample_method"]["major_axis"]["r_proj"];
        int max_total_points = config["scan_registration"]["sample_method"]["major_axis"]["max_total_points"];
        int azimuth_bins = config["scan_registration"]["sample_method"]["major_axis"]["azimuth_bins"];
        int elevation_bins = config["scan_registration"]["sample_method"]["major_axis"]["elevation_bins"];
        int min_points_per_bin = config["scan_registration"]["sample_method"]["major_axis"]["min_points_per_bin"];
        int max_points_per_bin = config["scan_registration"]["sample_method"]["major_axis"]["max_points_per_bin"];
        std::string sampling_strategy = config["scan_registration"]["sample_method"]["major_axis"]["sampling_strategy"];
        majorAxisSampling(pcl_cloud, last_pcl_cloud, candidate_indices, r, r_proj, max_total_points, azimuth_bins, elevation_bins, min_points_per_bin, max_points_per_bin, sampling_strategy, sampled_indices);
    }
    else
    {
        std::cerr << "Invalid SAMPLE_METHOD!" << std::endl;
    }
}


void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited)
    {
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;
    TicToc t_step;
    
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

    std::string timestamp = std::to_string(laserCloudMsg->header.stamp.toSec());
    std::string timesFile = OUTPUT_DIR + "scan_registration_times.txt";

    std::ofstream file(timesFile, std::ios::app);
    if (file.is_open())
    {
        file << "Frame time: " << timestamp << std::endl;  // 写入时间戳
        file.close();
    }
    else
    {
        std::cerr << "Unable to open times file for logging!" << std::endl;
        return;
    }


    /* -------------------- 1. downsample, remove nan and set scans -------------------- */
    
    t_step.tic();

    // pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudDownsampled(new pcl::PointCloud<pcl::PointXYZ>());
    // pcl::VoxelGrid<pcl::PointXYZ> downSizeFilter;
    // downSizeFilter.setInputCloud(laserCloudIn.makeShared());
    // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    // downSizeFilter.filter(*laserCloudDownsampled);

    // // 使用下采样后的点云进行后续处理
    // laserCloudIn = *laserCloudDownsampled;

    std::vector<int> indices;

    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE, MAXIMUM_RANGE);

    // // 将 PCL 点云数据转换为 CSF 格式
    // std::vector<csf::Point> csfPoints;
    // for (const auto &pt : laserCloudIn.points) {
    //     csf::Point csfPoint;
    //     csfPoint.x = pt.x;
    //     csfPoint.y = pt.y;
    //     csfPoint.z = pt.z;
    //     csfPoints.push_back(csfPoint);
    // }

    // // 设置 CSF 滤波器
    // CSF csf;
    // csf.setPointCloud(csfPoints);
    // csf.params.bSloopSmooth = true; // 可根据配置调整
    // csf.params.cloth_resolution = 0.5;
    // csf.params.interations = 500;
    // csf.params.rigidness = 3;
    // csf.params.class_threshold = 0.5;
    // csf.params.time_step = 0.65;

    // // 执行 CSF 滤波
    // std::vector<int> groundIndexes, offGroundIndexes;
    // csf.do_filtering(groundIndexes, offGroundIndexes, false);

    // // 提取地面点云
    // pcl::PointCloud<pcl::PointXYZ>::Ptr groundCloud(new pcl::PointCloud<pcl::PointXYZ>());
    // for (int idx : groundIndexes) {
    //     groundCloud->points.push_back(laserCloudIn.points[idx]);
    // }

    // // 将地面点云赋值给 laserCloudIn，用于后续处理
    // laserCloudIn = *groundCloud;


    // set scans
    int cloudSize = laserCloudIn.points.size();
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }

    bool halfPassed = false;
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    float upperBound, lowerBound;

    if (N_SCANS == 32)
    {
        upperBound = 15.0f;
        lowerBound = -25.0f;
    }
    else if (N_SCANS == 64)
    {
        upperBound = 2.0f;
        lowerBound = -24.33f;
    }

    int width = int(360 / AZIMUTH_RESOLUTION);
    RangeImage rangeImage(width, N_SCANS, upperBound, lowerBound);
    Eigen::MatrixXf range_image = Eigen::MatrixXf::Constant(N_SCANS, width, std::numeric_limits<float>::infinity());
    std::string data_format = config["scan_registration"]["compute_normal_method"]["format"];

    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        float range = sqrt(point.x * point.x + point.y * point.y);
        float vertical_angle = atan(point.z / range); // vertical angle (rad)
        float angle = vertical_angle * 180 / M_PI;
        int scanID = 0;

        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            // VLP32C
    		std::vector<float> scanAngles = {
                -25.000, -15.639, -11.310, -8.843, -7.254, -6.148, -5.333, -4.667, -4.000, 
                -3.667, -3.333, -3.000, -2.667, -2.333, -2.000, -1.667, -1.333, -1.000, 
                -0.667, -0.333, 0.000, 0.333, 0.667, 1.000, 1.333, 1.667, 2.333
            };

            // HDL32
            // std::vector<float> scanAngles = {
            //     -30.67, -29.33, -28.00, -26.66, -25.33, -24.00, -22.67, -21.33, -20.00, -18.67, 
            //     -17.33, -16.00, -14.67, -13.33, -12.00, -10.67, -9.33, -8.00, -6.66, -5.33, 
            //     -4.00, -2.67, -1.33, 0.00, 1.33, 2.67, 4.00, 5.33, 6.67, 8.00, 9.33, 10.67
            // };

            float min_diff = std::numeric_limits<float>::max();
    		for (int j = 0; j < scanAngles.size(); j++) 
            {
                float diff = std::abs(angle - scanAngles[j]);
                if (diff < min_diff) 
                {
                    min_diff = diff;
                    scanID = j;  // 找到最接近的scanID
                }
            }

            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((upperBound - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > upperBound || angle < lowerBound || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);

        float ori = -atan2(point.y, point.x); // azimuth angle (rad)
        if (!halfPassed)
        { 
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        float relTime = (ori - startOri) / (endOri - startOri);
        point.intensity = scanID + scanPeriod * relTime;
        laserCloudScans[scanID].push_back(point);

        if (data_format=="range_image") // Convert to range image
        {
            // 计算该点对应的图像列和行索引
            int col = static_cast<int>(relTime * width);
            int row = scanID;

            // 确保索引在范围内
            col = std::clamp(col, 0, width - 1);
            row = std::clamp(row, 0, N_SCANS - 1);

            // 更新深度图像中对应位置的值
            range_image(row, col) = std::min(range_image(row, col), range);
        }
    }
    
    cloudSize = count;
    printf("points size %d \n", cloudSize);

    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5;
		*laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
    }

    if(config["scan_registration"]["presample_method"]["method"]=="curvature")
    {
        int window_size = config["scan_registration"]["presample_method"]["curvature"]["window_size"];
        
        for (int i = 0; i < N_SCANS; i++) {
            for (int j = scanStartInd[i]; j < scanEndInd[i]; j++) {
                // 获取当前点
                PointType& point = laserCloud->points[j];

                // 判断当前点是否有足够的邻域点用于计算曲率
                if (j >= window_size && j < laserCloud->size() - window_size) {
                    float diffX = 0.0f, diffY = 0.0f, diffZ = 0.0f;

                    // 遍历当前点的邻域
                    for (int k = -window_size; k <= window_size; ++k) {
                        int neighbor_idx = j + k;

                        // 确保邻域点索引有效
                        if (neighbor_idx < 0 || neighbor_idx >= laserCloud->size()) {
                            continue;
                        }

                        // 获取邻居点
                        PointType& neighbor_point = laserCloud->points[neighbor_idx];

                        // 计算当前点与邻居点之间的差值
                        diffX += neighbor_point.x - point.x;
                        diffY += neighbor_point.y - point.y;
                        diffZ += neighbor_point.z - point.z;
                    }

                    // 计算曲率：平方和
                    float curvature = diffX * diffX + diffY * diffY + diffZ * diffZ;
                    
                    // 将计算得到的曲率赋值给当前点的`curvature`字段
                    point.curvature = curvature;  // 直接使用PointType中的curvature字段
                } else {
                    // 对于不满足窗口大小的点，直接赋曲率为0
                    point.curvature = 0.0f;
                }
            }
        }
    }

    t_step.tocAndLog("1. Preprocessing", timesFile);  // 记录时间


    /* -------------------- 2. compute normal: pointcloud(pca or cross_product) or range_image(FALS or SRI) --------------------*/

    t_step.tic();

    std::vector<int> filteredIndices;
    pcl::PointCloud<PointType>::Ptr filteredLaserCloud(new pcl::PointCloud<PointType>());

    std::vector<int> scanStartInd_filtered(N_SCANS, 0);
    std::vector<int> scanEndInd_filtered(N_SCANS, 0);

    Eigen::MatrixXf eigenvalues_matrix;  // 不指定初始大小
    Eigen::MatrixXf eigenvectors_matrix;
    int point_idx = 0;

    std::string compute_normal_method = config["scan_registration"]["compute_normal_method"]["method"];
    bool use_all_points = config["scan_registration"]["model"]["use_all_points"];
    std::vector<int> invalid_indices; // 不符合平面约束检查的点

    if (data_format=="pointcloud")
    {
        if (compute_normal_method=="pca")
        {
            int window_size = config["scan_registration"]["compute_normal_method"]["pca"]["window_size"];
            int iter_step = config["scan_registration"]["compute_normal_method"]["pca"]["iter_step"];
            float knn_distance_threshold = config["scan_registration"]["compute_normal_method"]["pca"]["knn_distance_threshold"];
            std::string neighbor_scan = config["scan_registration"]["compute_normal_method"]["pca"]["neighbor_scan"];
            float distance_threshold = config["scan_registration"]["compute_normal_method"]["pca"]["plane_constraint"]["distance_threshold"];
            float valid_points_threshold = config["scan_registration"]["compute_normal_method"]["pca"]["plane_constraint"]["valid_points_threshold"];
            int pca_failure = 0;

            // build kdtree for each scan
            std::vector<pcl::KdTreeFLANN<PointType>::Ptr> kdtrees(N_SCANS);
            if (neighbor_scan == "kdtree") {
                for (int i = 0; i < N_SCANS; ++i)
                {
                    kdtrees[i] = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
                    if (!laserCloudScans[i].points.empty())
                    {
                        kdtrees[i]->setInputCloud(laserCloudScans[i].makeShared());
                    }
                }
            }

            // PCA
            for (int i = 1; i < N_SCANS - 1; i++)
            {
                if (laserCloudScans[i].points.empty())
                    continue;
                if (scanEndInd[i] - scanStartInd[i] < 6 || scanEndInd[i-1] - scanStartInd[i-1] < 6 || scanEndInd[i+1] - scanStartInd[i+1] < 6)
                    continue;
                scanStartInd_filtered[i] = filteredLaserCloud->points.size() + 5;
                
                for (int j = 5; j < laserCloudScans[i].points.size() - 5; j++)
                {
                    float lambda1, lambda2, lambda3;
                    Eigen::Matrix3f eigen_vectors;
                    computeNormalPCA(laserCloudScans, i, j, lambda1, lambda2, lambda3, eigen_vectors, kdtrees,
                    window_size, iter_step, knn_distance_threshold, neighbor_scan, distance_threshold, valid_points_threshold);
                    
                    if (lambda1 == 0 && lambda2 == 0 && lambda3 == 0)
                    {
                        pca_failure++;
                        continue; // Skip if PCA computation was not successful
                    }
                    if (lambda1 == -1 && lambda2 == -1 && lambda3 == -1)
                    {
                        if (use_all_points)
                        {
                            invalid_indices.push_back(filteredLaserCloud->points.size());
                        }
                        else
                        {
                            continue;
                        }
                    }
                    
                    int global_idx = scanStartInd[i] + j; // Convert local idx to global idx

                    Eigen::Vector3f normal = eigen_vectors.col(2).normalized();
                    // 检查法向量和z轴的夹角，如果夹角大于90度，则取反
                    if (normal.dot(z_axis) < 0) {
                        normal = -normal;
                    }

                    eigenvalues_matrix.conservativeResize(3, point_idx + 1);
                    eigenvectors_matrix.conservativeResize(9, point_idx + 1);
                    eigenvalues_matrix.col(point_idx) << lambda1, lambda2, lambda3;
                    eigenvectors_matrix.col(point_idx) << eigen_vectors(0, 0), eigen_vectors(1, 0), eigen_vectors(2, 0),
                                                        eigen_vectors(0, 1), eigen_vectors(1, 1), eigen_vectors(2, 1),
                                                        eigen_vectors(0, 2), eigen_vectors(1, 2), eigen_vectors(2, 2);
                    point_idx++;

                    PointType filtered_point;
                    filtered_point.x = laserCloud->points[global_idx].x;
                    filtered_point.y = laserCloud->points[global_idx].y;
                    filtered_point.z = laserCloud->points[global_idx].z;
                    filtered_point.intensity = laserCloud->points[global_idx].intensity;

                    filtered_point.normal_x = normal.x();
                    filtered_point.normal_y = normal.y();
                    filtered_point.normal_z = normal.z();

                    filtered_point.curvature = laserCloud->points[global_idx].curvature;

                    filteredLaserCloud->push_back(filtered_point);
                    filteredIndices.push_back(global_idx);
                }
                scanEndInd_filtered[i] = filteredLaserCloud->points.size()-6;
            }
            printf("pca failure points size: %d \n", pca_failure);
            printf("plane check failure points size: %d \n", invalid_indices.size());
        }
        else if (compute_normal_method=="cross_product")
        {
            float knn_distance_threshold = config["scan_registration"]["compute_normal_method"]["cross_product"]["knn_distance_threshold"];
            std::string neighbor_scan = config["scan_registration"]["compute_normal_method"]["cross_product"]["neighbor_scan"];
            
            // build kdtree for each scan
            std::vector<pcl::KdTreeFLANN<PointType>::Ptr> kdtrees(N_SCANS);
            if (neighbor_scan == "kdtree") {
                for (int i = 0; i < N_SCANS; ++i)
                {
                    kdtrees[i] = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
                    if (!laserCloudScans[i].points.empty())
                    {
                        kdtrees[i]->setInputCloud(laserCloudScans[i].makeShared());
                    }
                }
            }

            for (int i = 1; i < N_SCANS - 1; i++)
            {
                if (laserCloudScans[i].points.empty())
                    continue;
                if (scanEndInd[i] - scanStartInd[i] < 6 || scanEndInd[i-1] - scanStartInd[i-1] < 6 || scanEndInd[i+1] - scanStartInd[i+1] < 6)
                    continue;
                scanStartInd_filtered[i] = filteredLaserCloud->points.size() + 5;
                
                for (int j = 5; j < laserCloudScans[i].points.size() - 5; j++)
                {
                    Eigen::Vector3f normal(0.0f, 0.0f, 0.0f);
                    computeNormalCrossProduct(laserCloudScans, i, j, kdtrees, knn_distance_threshold, neighbor_scan, normal);
                    
                    if (normal.isZero())
                    {
                        continue;
                    }
                    
                    int global_idx = scanStartInd[i] + j; // Convert local idx to global idx

                    // 检查法向量和z轴的夹角，如果夹角大于90度，则取反
                    if (normal.dot(z_axis) < 0) {
                        normal = -normal;
                    }

                    PointType filtered_point;
                    filtered_point.x = laserCloud->points[global_idx].x;
                    filtered_point.y = laserCloud->points[global_idx].y;
                    filtered_point.z = laserCloud->points[global_idx].z;
                    filtered_point.intensity = laserCloud->points[global_idx].intensity;

                    filtered_point.normal_x = normal.x();
                    filtered_point.normal_y = normal.y();
                    filtered_point.normal_z = normal.z();

                    filtered_point.curvature = laserCloud->points[global_idx].curvature;

                    filteredLaserCloud->push_back(filtered_point);
                    filteredIndices.push_back(global_idx);
                }
                scanEndInd_filtered[i] = filteredLaserCloud->points.size()-6;
            }
        }
        else
        {
            std::cerr << "Invalid COMPUTE_NORMAL_METHOD!" << std::endl;
        }
    }
    else if (data_format=="range_image")
    {
        std::vector<Eigen::Vector3f> normal_list;
        if (compute_normal_method=="FALS")
        {
            // window size need to divide 2 and round down
            // int window_size = int(config["scan_registration"]["compute_normal_method"]["FALS"]["window_size"] / 2);
            int window_size = config["scan_registration"]["compute_normal_method"]["FALS"]["window_size"];
            
            if(rangeImage.computeNormalFALS(range_image, window_size, filteredIndices, normal_list))
            {
                for (int i = 0; i < filteredIndices.size() - 1; i++)
                {
                    // 检查法向量和z轴的夹角，如果夹角大于90度，则取反
                    if (normal_list[i].dot(z_axis) < 0) {
                        normal_list[i] = -normal_list[i];
                    }

                    PointType filtered_point;
                    filtered_point.x = laserCloud->points[filteredIndices[i]].x;
                    filtered_point.y = laserCloud->points[filteredIndices[i]].y;
                    filtered_point.z = laserCloud->points[filteredIndices[i]].z;
                    filtered_point.intensity = laserCloud->points[filteredIndices[i]].intensity;

                    filtered_point.normal_x = normal_list[i].x();
                    filtered_point.normal_y = normal_list[i].y();
                    filtered_point.normal_z = normal_list[i].z();

                    filtered_point.curvature = laserCloud->points[filteredIndices[i]].curvature;

                    filteredLaserCloud->push_back(filtered_point);
                }
            }
            else
            {
                std::cerr << "Compute normal using FALS failed!" << std::endl;
            }
        }
        else if (compute_normal_method=="SRI")
        {
            int window_size = config["scan_registration"]["compute_normal_method"]["SRI"]["window_size"];
            if(rangeImage.computeNormalSRI(range_image, window_size, filteredIndices, normal_list))
            {
                for (int i = 0; i < filteredIndices.size() - 1; i++)
                {
                    // 检查法向量和z轴的夹角，如果夹角大于90度，则取反
                    if (normal_list[i].dot(z_axis) < 0) {
                        normal_list[i] = -normal_list[i];
                    }

                    PointType filtered_point;
                    filtered_point.x = laserCloud->points[filteredIndices[i]].x;
                    filtered_point.y = laserCloud->points[filteredIndices[i]].y;
                    filtered_point.z = laserCloud->points[filteredIndices[i]].z;
                    filtered_point.intensity = laserCloud->points[filteredIndices[i]].intensity;

                    filtered_point.normal_x = normal_list[i].x();
                    filtered_point.normal_y = normal_list[i].y();
                    filtered_point.normal_z = normal_list[i].z();

                    filtered_point.curvature = laserCloud->points[filteredIndices[i]].curvature;

                    filteredLaserCloud->push_back(filtered_point);
                }
            }
            else
            {
                std::cerr << "Compute normal using SRI failed!" << std::endl;
            }
        }
        else
        {
            std::cerr << "Invalid COMPUTE_NORMAL_METHOD!" << std::endl;
        }
    }
    else
    {
        std::cerr << "Invalid DATA_FORMAT!" << std::endl;
    }
    
    printf("Filtered points size: %d \n", filteredLaserCloud->points.size());

    // visualization
    // add marker
	visualization_msgs::Marker pcaMarker;
	pcaMarker.header.frame_id = "/camera_init";
	pcaMarker.ns = "pca_features";
	pcaMarker.type = visualization_msgs::Marker::LINE_LIST;
	pcaMarker.action = visualization_msgs::Marker::ADD;
	pcaMarker.scale.x = 0.02;
	pcaMarker.pose.orientation.w = 1.0;
	pcaMarker.header.stamp = laserCloudMsg->header.stamp;
    visualizePCAFeatures(filteredLaserCloud, pcaMarker);
	pubPCAFeatures.publish(pcaMarker);

    saveMarkerToFile(pcaMarker, OUTPUT_DIR + "pca_markers/" + timestamp + ".obj");
    // savePointCloudToTxt(filteredLaserCloud, OUTPUT_DIR + "filteredpts/" + timestamp + ".txt"); 

    t_step.tocAndLog("2. Compute normal", timesFile); 


    /* -------------------- 3. Sampling --------------------*/
    t_step.tic();

    std::vector<int> candidate_indices;
    std::vector<int> sampled_indices;

    // Convert to pcl::PointXYZINormal for publishing
    pcl::PointCloud<PointType>::Ptr pcl_cloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr pcl_surface_cloud(new pcl::PointCloud<PointType>());

    DP dp_cloud;

    if (config["scan_registration"]["presample_method"]["method"]=="tensor_voting")
    {
        // Tensor Voting
        int k = config["scan_registration"]["presample_method"]["tensor_voting"]["k"];
        float sigma = config["scan_registration"]["presample_method"]["tensor_voting"]["sigma"];

        if (eigenvalues_matrix.rows() > 0 && eigenvalues_matrix.cols() > 0 && eigenvectors_matrix.rows() > 0 && eigenvectors_matrix.cols() > 0)
        {
            dp_cloud = applySaliencyFilter(filteredLaserCloud, eigenvalues_matrix, eigenvectors_matrix, k, sigma);
        }
        else
        {
            std::cerr << "Eigenvalues and eigenvectors matrices are not initialized! Choose pca to compute normal." << std::endl;
        }

        // Reverse normals if necessary
        for (int i = 0; i < dp_cloud.features.cols(); ++i) {
            /* ---------- ATTENTION: LIBPOINTMATCHER HAS BUGS, HERE NORMAL IS ACTUALLY TANGENT IN LIBPOINTMATCHER ----------*/
            // 获取法向量，位于 descriptors 的第 6 到 8 行
            Eigen::Vector3f normal = dp_cloud.descriptors.block<3,1>(6, i);

            // 反转法向量，如果和 z 轴夹角大于 90 度
            if (normal.dot(z_axis) < 0) {
                dp_cloud.descriptors.block<3,1>(6, i) = -normal;
            }
        }

        // 提取非球点（非ball点）作为候选点
        for (int i = 0; i < dp_cloud.features.cols(); ++i) {
            // labels 在 descriptors 的第 9 行
            if (dp_cloud.descriptors(9, i) != 1) {
                candidate_indices.push_back(i);
            }
        }

        // To pcl::PointType
        pcl_cloud = convertDataPointsToPointCloud(dp_cloud);
        
    }
    else if (config["scan_registration"]["presample_method"]["method"]=="geometric_features")
    {
        Eigen::MatrixXf features; 
        float planarity_threshold = config["scan_registration"]["presample_method"]["geometric_features"]["planarity_threshold"];
        if (eigenvalues_matrix.rows() > 0 && eigenvalues_matrix.cols() > 0) {
            computeGeometricFeatures(eigenvalues_matrix, features, planarity_threshold, candidate_indices);
        }
        else
        {
            std::cerr << "Eigenvalues matrix is not initialized! Choose pca to compute normal." << std::endl;
        }

        pcl_cloud = filteredLaserCloud;
    }
    else if(config["scan_registration"]["presample_method"]["method"]=="curvature")
    {
        float curvature_threshold = config["scan_registration"]["presample_method"]["curvature"]["curvature_threshold"];
        candidate_indices.clear();
        for (int i = 0; i < filteredLaserCloud->points.size(); ++i) {
            if (filteredLaserCloud->points[i].curvature > curvature_threshold) {
                candidate_indices.push_back(i);  // 满足条件，加入候选点索引
            }
        }

        pcl_cloud = filteredLaserCloud;
    }
    else
    {
        std::cerr << "Invalid PRESAMPLE_METHOD!" << std::endl;
    }

    std::string sample_method = config["scan_registration"]["sample_method"]["method"]; // three_axis, random, normal, major_axis

    if (use_all_points && invalid_indices.size() > 0)
    {
        candidate_indices.erase(
            std::remove_if(candidate_indices.begin(), candidate_indices.end(),
                        [&invalid_indices](int idx) {
                            return std::find(invalid_indices.begin(), invalid_indices.end(), idx) != invalid_indices.end();
                        }),
            candidate_indices.end());
    }

    printf("Presampled points size: %d \n", candidate_indices.size());

    t_step.tocAndLog("3. Presampling", timesFile);
    t_step.tic();

    samplePointCloud(sample_method, pcl_cloud, eigenvalues_matrix, candidate_indices, frame, sampled_indices);

    // use indices to sample pointcloud
    for (int idx : sampled_indices) {
        pcl_surface_cloud->points.push_back(pcl_cloud->points[idx]);
    }

    frame++;
    last_pcl_cloud = pcl_cloud;

    printf("pcl_cloud size: %d \n", pcl_cloud->points.size());
    printf("pcl_surface_cloud size: %d \n", pcl_surface_cloud->points.size());

    // Publish point cloud and surface cloud
    publishPointCloud(pcl_cloud, pubLaserCloud, laserCloudMsg->header.stamp);
    publishPointCloud(pcl_surface_cloud, pubPointsFlat, laserCloudMsg->header.stamp);

    savePointCloudToTxt(pcl_cloud, OUTPUT_DIR + "pcl_cloud/" + timestamp + ".txt");
    savePointCloudToTxt(pcl_surface_cloud, OUTPUT_DIR + "pcl_surface_cloud/" + timestamp + ".txt");


    if (config["scan_registration"]["presample_method"]["method"]=="tensor_voting"){
        DP dp_surface_cloud(dp_cloud.createSimilarEmpty());
        for (int i = 0; i < sampled_indices.size(); ++i) {
            dp_surface_cloud.setColFrom(i, dp_cloud, sampled_indices[i]);
        }
        dp_surface_cloud.conservativeResize(sampled_indices.size());

        // Publish as Datapoints
        publishDataPointsAsPointCloud2(dp_cloud, laserCloudMsg->header.frame_id, pubLaserCloudDP, laserCloudMsg->header.stamp);
        publishDataPointsAsPointCloud2(dp_surface_cloud, laserCloudMsg->header.frame_id, pubPointsFlatDP, laserCloudMsg->header.stamp);

        // saveCloudFeaturesAndDescriptors(dp_cloud, OUTPUT_DIR + "dp_cloud/" + timestamp + ".txt");
        // saveCloudFeaturesAndDescriptors(dp_surface_cloud, OUTPUT_DIR + "dp_surface_cloud/" + timestamp + ".txt");
    }

    // add marker
	visualization_msgs::Marker pcaMarker_2;
	pcaMarker_2.header.frame_id = "/camera_init";
	pcaMarker_2.ns = "pca_features_2";
	pcaMarker_2.type = visualization_msgs::Marker::LINE_LIST;
	pcaMarker_2.action = visualization_msgs::Marker::ADD;
	pcaMarker_2.scale.x = 0.02;
	pcaMarker_2.pose.orientation.w = 1.0;
	pcaMarker_2.header.stamp = laserCloudMsg->header.stamp;
    visualizePCAFeatures(pcl_cloud, pcaMarker_2);
    saveMarkerToFile(pcaMarker_2, OUTPUT_DIR + "pca_markers_2/" + timestamp + ".obj");

    visualization_msgs::Marker pcaMarker_2_surface;
	pcaMarker_2_surface.header.frame_id = "/camera_init";
	pcaMarker_2_surface.ns = "pca_features_2_surface";
	pcaMarker_2_surface.type = visualization_msgs::Marker::LINE_LIST;
	pcaMarker_2_surface.action = visualization_msgs::Marker::ADD;
	pcaMarker_2_surface.scale.x = 0.02;
	pcaMarker_2_surface.pose.orientation.w = 1.0;
	pcaMarker_2_surface.header.stamp = laserCloudMsg->header.stamp;
    visualizePCAFeatures(pcl_surface_cloud, pcaMarker_2_surface);
    saveMarkerToFile(pcaMarker_2_surface, OUTPUT_DIR + "pca_markers_2_surface/" + timestamp + ".obj");

    t_step.tocAndLog("3. Sampling", timesFile);


    t_whole.tocAndLog("Total time", timesFile);

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "scan_registration");
    ros::NodeHandle nh;

    try {
        loadConfig();
    } catch (const std::exception& e) {
        std::cerr << "加载配置文件失败: " << e.what() << std::endl;
        return -1;
    }

    nh.param<int>("scan_line", N_SCANS, 16);

    nh.param<float>("azimuth_resolution", AZIMUTH_RESOLUTION, 0.9);

    nh.param<float>("minimum_range", MINIMUM_RANGE, 0.1);

    nh.param<float>("maximum_range", MAXIMUM_RANGE, 120);

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

	// ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/kitti/velo/pointcloud", 100, laserCloudHandler);
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
	
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_filtered", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    pubPCAFeatures = nh.advertise<visualization_msgs::Marker>("/pca_features", 100);

    pubPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubLaserCloudDP = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_filtered_dp", 100);

    pubPointsFlatDP = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat_dp", 100);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}
