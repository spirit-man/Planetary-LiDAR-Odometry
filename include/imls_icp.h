#ifndef IMLS_ICP_H
#define IMLS_ICP_H

#include <iostream>
#include <vector>
#include <set>
#include <nabo/nabo.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/unsupported/Eigen/Polynomials>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sys/time.h>
#include <math.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pointmatcher/PointMatcher.h>
#include <pointmatcher/DataPointsFilters/utils/sparsetv.h>
#include <ros/ros.h>
#include "common.h"


typedef PointMatcher<double> PMD;
typedef PMD::DataPoints DPD;

// 自定义哈希函数，用于 Eigen::Vector3d
struct Vector3dHash {
    std::size_t operator()(const Eigen::Vector3d& vec) const {
        std::size_t hx = std::hash<double>{}(vec.x());
        std::size_t hy = std::hash<double>{}(vec.y());
        std::size_t hz = std::hash<double>{}(vec.z());
        return hx ^ (hy << 1) ^ (hz << 2); // 合并哈希值
    }
};

// 自定义相等比较函数
struct Vector3dEqual {
    bool operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const {
        return v1.isApprox(v2); // 使用 Eigen 的 isApprox 函数来比较double
    }
};

class IMLSICPMatcher
{
public:
    IMLSICPMatcher();
    IMLSICPMatcher(int _iter, double _h, double _r, double _r_normal, double _r_proj, 
                    bool _useTensorVoting, bool _isGetNormals, bool _useProjectedDistance, 
                    int _tensor_k, double _tensor_sigma, double _tensor_distance_threshold,
                    int _search_number_normal, int _search_number, bool _normal_angle_constraint,
                    double _angle_diff_threshold, const std::string& _output_dir);
    ~IMLSICPMatcher();

    void setSourcePointCloud(pcl::PointCloud<PointType>::Ptr cloud);

    void setTargetPointCloud(pcl::PointCloud<PointType>::Ptr cloud);

    void setTargetPointCloudDP(const DP& cloud);

    void setParameters(int _iter, double _h, double _r, double _r_normal, double _r_proj, 
                   bool _useTensorVoting, bool _isGetNormals, bool _useProjectedDistance, 
                   int _tensor_k, double _tensor_sigma, double _tensor_distance_threshold,
                   int _search_number_normal, int _search_number, bool _normal_angle_constraint,
                   double _angle_diff_threshold, const std::string& _output_dir);
                   

    std::pair<TensorVoting<double>, std::vector<std::size_t>> VoteForAny(TensorVoting<double>& tv_input, TensorVoting<double>& tv_output, const DPD& inputPts, DPD& outputPts);

    void createKDTreeUsingLocalMap(void );


    //IMLS函数，主要功能为进行把xi投影到表面上．
    bool ImplicitMLSFunction(PointType& x,
                             double& height);

    //把点云进行投影，投影到表面surface之中．
    void ProjSourcePtToSurface(pcl::PointCloud<PointType>::Ptr &in_cloud,
                                pcl::PointCloud<PointType>::Ptr &out_cloud,
                                const std::string &timestamp, 
                                const int &i);

    Eigen::Vector3d ComputeNormal(std::vector<Eigen::Vector3d>& nearPoints);

    bool Match(Eigen::Matrix4d& finalPose,
               Eigen::Matrix4d& covariance,
               const std::string& timestamp);

private:
    void RemoveNANandINFData(pcl::PointCloud<PointType>::Ptr &cloud);

    // 目标点云和当前点云，目标点云为参考点云．
    pcl::PointCloud<PointType>::Ptr m_sourcePointCloud,m_targetPointCloud;
    DPD m_targetPointCloudDP;

    // 所有的激光帧数据，每一个激光帧数据对应的点云的下标．
    std::map<int,std::vector<int> > m_LaserFrames;

    // 指针
    Nabo::NNSearchD* m_pSourceKDTree;
    Nabo::NNSearchD* m_pTargetKDTree;

    Eigen::MatrixXd m_sourceKDTreeDataBase;
    Eigen::MatrixXd m_targetKDTreeDataBase;


    // 点云的id．
    int m_PtID;

    // LocalMap和Tree之间的下标偏移，Tree永远是从0开始．
    int m_offsetBetweenLocalMapAndTree;

    // 迭代次数．
    int m_iterations;

    // 含义见论文．用来计算权重．
    // m_r ~ 3*m_h
    double m_h, m_r, m_r_normal, m_r_proj;

    // 是否使用Tensor Voting
    bool m_useTensorVoting;

    // 判断使用传入的法向量还是重新计算．
    bool m_isGetNormals;

    // 是否使用法向量投影距离找最近
    bool m_useProjectedDistance;

    // Tensor Voting 相关参数
    int m_tensor_k;
    double m_tensor_sigma, m_tensor_distance_threshold;

    // 重新计算法向量的搜索点数
    int m_search_number_normal;
    
    // ImplicitMLSFunction 中搜索的隐式点数量
    int m_search_number;

    // 法向量连续性检查
    bool m_normal_angle_constraint;

    double m_angle_diff_threshold;

    // 结果输出路径
    std::string m_output_dir;
};


#endif // IMLS_ICP_H
