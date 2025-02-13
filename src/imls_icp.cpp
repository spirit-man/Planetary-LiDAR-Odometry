#include "imls_icp.h"
#include "saver.h"
#include "common.h"
#include "solver.h"
#include <unordered_map>
#include <functional>


IMLSICPMatcher::IMLSICPMatcher(void )
{
    m_PtID = 0;
    m_pTargetKDTree = m_pSourceKDTree = NULL;

    m_iterations = 30;
    m_h = 0.2;
    m_r = 0.6;
    m_r_normal = 0.2;
    m_r_proj = 0.6;
    m_useTensorVoting = false;
    m_isGetNormals = true;
    m_useProjectedDistance = true;
    m_tensor_k = 50;
    m_tensor_sigma = 0.2;
    m_tensor_distance_threshold = 0.6;
    m_search_number_normal = 10;
    m_search_number = 20;
    m_normal_angle_constraint = false;
    m_angle_diff_threshold = 60;
    m_output_dir = "./";
}

//构造函数．
IMLSICPMatcher::IMLSICPMatcher(int _iter, double _h, double _r, double _r_normal, double _r_proj, 
                    bool _useTensorVoting, bool _isGetNormals, bool _useProjectedDistance, 
                    int _tensor_k, double _tensor_sigma, double _tensor_distance_threshold,
                    int _search_number_normal, int _search_number, bool _normal_angle_constraint,
                    double _angle_diff_threshold, const std::string& _output_dir)
{
    IMLSICPMatcher matcher;
    matcher.setParameters(_iter, _h, _r, _r_normal, _r_proj, 
                   _useTensorVoting, _isGetNormals, _useProjectedDistance, 
                   _tensor_k, _tensor_sigma, _tensor_distance_threshold,
                   _search_number_normal, _search_number, _normal_angle_constraint,
                   _angle_diff_threshold, _output_dir);
}

//析构函数，释放内存．
IMLSICPMatcher::~IMLSICPMatcher(void )
{
    if(m_pTargetKDTree != NULL)
        delete m_pTargetKDTree;

    if(m_pSourceKDTree != NULL)
        delete m_pSourceKDTree;
}

//去除非法数据
void IMLSICPMatcher::RemoveNANandINFData(pcl::PointCloud<PointType>::Ptr &cloud)
{
    pcl::PointCloud<PointType>::iterator it = cloud->begin();
    while (it != cloud->end())
    {
        if (!pcl::isFinite<PointType>(*it)) // 使用 PCL 提供的函数检查点的有效性
        {
            it = cloud->erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void IMLSICPMatcher::setSourcePointCloud(pcl::PointCloud<PointType>::Ptr cloud)
{
    m_sourcePointCloud = cloud;
    RemoveNANandINFData(m_sourcePointCloud);
}

void IMLSICPMatcher::setTargetPointCloud(pcl::PointCloud<PointType>::Ptr cloud)
{
    m_targetPointCloud = cloud;
    RemoveNANandINFData(m_targetPointCloud);

    if(m_pTargetKDTree != NULL)
    {
        delete m_pTargetKDTree;
        m_pTargetKDTree = NULL;
    }

    //构建kd树．
    if(m_pTargetKDTree == NULL)
    {
        m_targetKDTreeDataBase.resize(3,m_targetPointCloud->size());
        for(int i = 0; i < m_targetPointCloud->size();i++)
        {
            m_targetKDTreeDataBase(0,i) = (*m_targetPointCloud)[i].x;
            m_targetKDTreeDataBase(1,i) = (*m_targetPointCloud)[i].y;
            m_targetKDTreeDataBase(2,i) = (*m_targetPointCloud)[i].z;
        }
        m_pTargetKDTree = Nabo::NNSearchD::createKDTreeLinearHeap(m_targetKDTreeDataBase);
    }
}

void IMLSICPMatcher::setTargetPointCloudDP(const DP& dp_cloud)
{
    // m_targetPointCloudDP = dp_cloud;
    
    DPD::Labels featureLabels;
    DPD::Labels descriptorLabels;
    const int pointCount = dp_cloud.features.cols();
    for (const auto& label : dp_cloud.featureLabels)
    {
        featureLabels.push_back(PMD::DataPoints::Label(label.text, label.span));
    }
    for (const auto& label : dp_cloud.descriptorLabels)
    {
        descriptorLabels.push_back(PMD::DataPoints::Label(label.text, label.span));
    }

    DPD dpd_cloud(featureLabels, descriptorLabels, pointCount);
    dpd_cloud.features = dp_cloud.features.cast<double>();
    dpd_cloud.descriptors = dp_cloud.descriptors.cast<double>();

    m_targetPointCloudDP = dpd_cloud;

    if(m_pTargetKDTree != NULL)
    {
        delete m_pTargetKDTree;
        m_pTargetKDTree = NULL;
    }

    if(m_pTargetKDTree == NULL)
    {
        m_targetKDTreeDataBase.resize(3, m_targetPointCloudDP.features.cols());
        for (int i = 0; i < m_targetPointCloudDP.features.cols(); ++i)
        {
            m_targetKDTreeDataBase(0, i) = m_targetPointCloudDP.features(0, i);
            m_targetKDTreeDataBase(1, i) = m_targetPointCloudDP.features(1, i);
            m_targetKDTreeDataBase(2, i) = m_targetPointCloudDP.features(2, i);
        }
        m_pTargetKDTree = Nabo::NNSearchD::createKDTreeLinearHeap(m_targetKDTreeDataBase);
    }
}

void IMLSICPMatcher::setParameters(int _iter, double _h, double _r, double _r_normal, double _r_proj, 
                                   bool _useTensorVoting, bool _isGetNormals, bool _useProjectedDistance, 
                                   int _tensor_k, double _tensor_sigma, double _tensor_distance_threshold,
                                   int _search_number_normal, int _search_number, bool _normal_angle_constraint,
                                   double _angle_diff_threshold, const std::string& _output_dir)
{
    m_iterations = _iter;
    m_h = _h;
    m_r = _r;
    m_r_normal = _r_normal;
    m_r_proj = _r_proj;
    m_useTensorVoting = _useTensorVoting;
    m_isGetNormals = _isGetNormals;
    m_useProjectedDistance = _useProjectedDistance;
    m_tensor_k = _tensor_k;
    m_tensor_sigma = _tensor_sigma;
    m_tensor_distance_threshold = _tensor_distance_threshold;
    m_search_number_normal = _search_number_normal;
    m_search_number = _search_number;
    m_normal_angle_constraint = _normal_angle_constraint;
    m_angle_diff_threshold = _angle_diff_threshold;
    m_output_dir = _output_dir;
}


std::pair<TensorVoting<double>, std::vector<std::size_t>> IMLSICPMatcher::VoteForAny(TensorVoting<double>& tv_input, TensorVoting<double>& tv_output, const DPD& inputPts, DPD& outputPts) {
    const std::size_t nbOutputPts = outputPts.getNbPoints();
    const std::size_t nbInputPts = inputPts.getNbPoints();

    std::size_t invalid_index_count = 0;
    std::size_t dist_out_of_range_count = 0;

    // Step 1: 对输入点云进行张量编码
    tv_input.encode(inputPts, TensorVoting<double>::Encoding::AWARE_TENSOR);

    std::vector<typename TensorVoting<double>::Tensor> inputTensors(nbInputPts);
    #pragma omp parallel for
    for (std::size_t i = 0; i < nbInputPts; ++i) {
        inputTensors[i] = tv_input.tensors(i);
    }

    // Step 2: 为每个目标点初始化张量
    tv_output.encode(outputPts, TensorVoting<double>::Encoding::ZERO);

    // Step 3: 使用 Nabo 库进行 KNN 查找
    typedef Nabo::NearestNeighbourSearch<double> NNS;
    Eigen::MatrixXi indices(tv_output.k, nbOutputPts);
    Eigen::MatrixXd dists(tv_output.k, nbOutputPts);

    Eigen::MatrixXd outputFeatures = outputPts.features.topRows(3).cast<double>();

    m_pTargetKDTree->knn(outputFeatures, indices, dists, tv_output.k);

    // Step 4: 对目标点进行投票
    for (std::size_t outputIdx = 0; outputIdx < nbOutputPts; ++outputIdx) {
        const typename TensorVoting<double>::Vector3 x_output = outputPts.features.block<3,1>(0, outputIdx);

        for (std::size_t j = 0; j < tv_output.k; ++j) {
            const std::size_t inputIdx = indices(j, outputIdx);
            if (inputIdx == NNS::InvalidIndex) {
                invalid_index_count++;  // 计数无效索引
                continue;
            }

            const typename TensorVoting<double>::Vector3 x_input = inputPts.features.block<3,1>(0, inputIdx);
            const typename TensorVoting<double>::Vector3 r_ig = x_output - x_input;
            const double dist = r_ig.norm() / tv_output.sigma;

            if (dist <= 0. || dist >= m_tensor_distance_threshold) {
                dist_out_of_range_count++;  // 计数距离超出范围
                continue;
            }

            const typename TensorVoting<double>::Vector3 r_ig_normalized = r_ig.normalized();
            const double weight = std::exp(-std::pow(r_ig.norm(), 2) / tv_output.sigma);
            const typename TensorVoting<double>::Tensor& T_input = inputTensors[inputIdx];
            const typename TensorVoting<double>::Tensor R_ij = Eigen::Matrix<double, 3, 3>::Identity() - 2 * r_ig_normalized * r_ig_normalized.transpose();
            const typename TensorVoting<double>::Tensor R_prime_ij = (Eigen::Matrix<double, 3, 3>::Identity() - 0.5 * r_ig_normalized * r_ig_normalized.transpose()) * R_ij;
            const typename TensorVoting<double>::Tensor S_ij = weight * R_ij * T_input * R_prime_ij;

            tv_output.tensors(outputIdx) += S_ij;
        }
    }

    // Step 5: 保留非零张量的索引
    std::vector<std::size_t> nonZeroIndices;
    for (std::size_t i = 0; i < tv_output.tensors.rows(); ++i) {
        if (!tv_output.tensors(i).isZero()) {
            nonZeroIndices.push_back(i);
        }
    }

    // Step 6: 创建非零张量的 tv_output_nonZero
    TensorVoting<double> tv_output_nonZero = tv_output;
    tv_output_nonZero.tensors.resize(nonZeroIndices.size(), 1);
    for (std::size_t i = 0; i < nonZeroIndices.size(); ++i) {
        tv_output_nonZero.tensors(i) = tv_output.tensors(nonZeroIndices[i]);
    }

    tv_output_nonZero.decompose();
    tv_output_nonZero.toDescriptors();

    std::cout << "Number of points skipped due to invalid index: " << invalid_index_count << std::endl;
    std::cout << "Number of points skipped due to distance out of range: " << dist_out_of_range_count << std::endl;

    // Step 7: 初始化 descriptors
    PMD::Matrix surfaceness = PMD::Matrix::Zero(1, nbOutputPts);
    PMD::Matrix curveness = PMD::Matrix::Zero(1, nbOutputPts);
    PMD::Matrix pointness = PMD::Matrix::Zero(1, nbOutputPts);
    PMD::Matrix normals = PMD::Matrix::Zero(3, nbOutputPts);
    PMD::Matrix tangents = PMD::Matrix::Zero(3, nbOutputPts);
    PMD::Matrix labels = PMD::Matrix::Zero(1, nbOutputPts);
    labels.setConstant(3);
    PMD::Matrix sticks = PMD::Matrix::Zero(4, nbOutputPts);
    PMD::Matrix plates = PMD::Matrix::Zero(7, nbOutputPts);
    PMD::Matrix balls = PMD::Matrix::Zero(1, nbOutputPts);

    // Step 8: 给 nonZeroIndices 赋值
    for (std::size_t i = 0; i < nonZeroIndices.size(); ++i) {
        std::size_t idx = nonZeroIndices[i];
        surfaceness(0, idx) = tv_output_nonZero.surfaceness(0, i);
        curveness(0, idx) = tv_output_nonZero.curveness(0, i);
        pointness(0, idx) = tv_output_nonZero.pointness(0, i);
        normals.col(idx) = tv_output_nonZero.normals.col(i);

        // Unify orentation
        /* ---------- ATTENTION: LIBPOINTMATCHER HAS BUGS, HERE NORMAL IS ACTUALLY TANGENT IN LIBPOINTMATCHER ----------*/
        Eigen::Vector3d z_axis(0, 0, 1);
        if (tv_output_nonZero.tangents.col(i).dot(z_axis) < 0) {
            tv_output_nonZero.tangents.col(i)= -tv_output_nonZero.tangents.col(i);
        }

        tangents.col(idx) = tv_output_nonZero.tangents.col(i);
        sticks.col(idx) = tv_output_nonZero.sticks.col(i);
        plates.col(idx) = tv_output_nonZero.plates.col(i);
        balls(0, idx) = tv_output_nonZero.balls(0, i);
    }

    // Step 9: 将 descriptors 添加到 outputPts 中
    outputPts.addDescriptor("surfaceness", surfaceness);
    outputPts.addDescriptor("curveness", curveness);
    outputPts.addDescriptor("pointness", pointness);
    outputPts.addDescriptor("normals", normals);
    outputPts.addDescriptor("tangents", tangents);
    outputPts.addDescriptor("labels", labels);
    outputPts.addDescriptor("sticks", sticks);
    outputPts.addDescriptor("plates", plates);
    outputPts.addDescriptor("balls", balls);

    return {tv_output_nonZero, nonZeroIndices};
}

//IMLS函数，主要用来进行曲面投影．
//可以认为是xi在曲面上的高度．
//用target_sourcePtcloud构造一个kd树．
bool IMLSICPMatcher::ImplicitMLSFunction(PointType& x,
                                         double& height)
{
    double weightSum = 0.0;
    double projSum = 0.0;
    int valid_number = m_search_number;

    //创建KD树
    if(m_pTargetKDTree == NULL)
    {
        m_targetKDTreeDataBase.resize(3,m_targetPointCloud->size());
        for(int i = 0; i < m_targetPointCloud->size();i++)
        {
            // if (std::round((*m_targetPointCloud)[i].intensity) == std::round(x.intensity))  // 仅处理相同 scan ID 的点
            // {
            //     m_targetKDTreeDataBase(0,i) = (*m_targetPointCloud)[i].x;
            //     m_targetKDTreeDataBase(1,i) = (*m_targetPointCloud)[i].y;
            //     m_targetKDTreeDataBase(2,i) = (*m_targetPointCloud)[i].z;
            // }
            m_targetKDTreeDataBase(0,i) = (*m_targetPointCloud)[i].x;
            m_targetKDTreeDataBase(1,i) = (*m_targetPointCloud)[i].y;
            m_targetKDTreeDataBase(2,i) = (*m_targetPointCloud)[i].z;
        }
        m_pTargetKDTree = Nabo::NNSearchD::createKDTreeLinearHeap(m_targetKDTreeDataBase);
    }

    // 找到位于点x附近(m_r)的所有的点云
    Eigen::VectorXi nearIndices(m_search_number);
    Eigen::VectorXd nearDist2(m_search_number);

    //找到某一个点的最近邻．
    //搜索m_search_number个最近邻
    //下标储存在nearIndices中，距离储存在nearDist2中．
    //最大搜索距离为m_r
    Eigen::Vector3d x_vec(x.x, x.y, x.z);
    Eigen::Vector3d xi_normalVector(x.normal_x, x.normal_y, x.normal_z);

    if (m_useProjectedDistance)
    {
        std::vector<std::pair<double, int>> projectionDistances;
        Eigen::Matrix3Xd diff = m_targetKDTreeDataBase.colwise() - x_vec; // 计算每个点与目标点的差值
        Eigen::Matrix3Xd cross_product = diff.colwise().cross(xi_normalVector);    // 计算叉乘
        Eigen::VectorXd proj_dist = cross_product.colwise().norm();      // 计算投影距离的模长

        for (int j = 0; j < proj_dist.size(); ++j) {
            if (diff.col(j).norm() < m_r_proj && proj_dist(j) < m_r) {
                projectionDistances.emplace_back(proj_dist(j), j);
            }
        }

        if (projectionDistances.size() > 0) 
        {
            // 按投影距离排序并选择最近的 m_search_number 个点
            std::sort(projectionDistances.begin(), projectionDistances.end());
            valid_number = std::min(m_search_number, static_cast<int>(projectionDistances.size()));
            projectionDistances.resize(valid_number);
            nearIndices.resize(valid_number);
            nearDist2.resize(valid_number);

            for (size_t i = 0; i < valid_number; ++i) {
                nearIndices(i) = projectionDistances[i].second;
                nearDist2(i) = projectionDistances[i].first * projectionDistances[i].first; // 存储距离的平方
            }
        }
        else
        {
            return false;
        }
    }
    else
    {
        m_pTargetKDTree->knn(x_vec,nearIndices,nearDist2,m_search_number,0,
                            Nabo::NNSearchD::SORT_RESULTS | Nabo::NNSearchD::ALLOW_SELF_MATCH|
                            Nabo::NNSearchD::TOUCH_STATISTICS,
                            m_r);
    }

    std::vector<Eigen::Vector3d> nearPoints;
    std::vector<Eigen::Vector3d> nearNormals;
    for(int i = 0; i < valid_number;i++)
    {
        //说明最近邻是合法的．
        if(nearDist2(i) < std::numeric_limits<double>::infinity() &&
                std::isinf(nearDist2(i)) == false &&
                std::isnan(nearDist2(i)) == false)
        {
            //该最近邻在原始数据中的下标．
            int index = nearIndices(i);

            // if (std::round((*m_targetPointCloud)[index].intensity) != std::round(x.intensity))
            //     continue;

            Eigen::Vector3d tmpPt(m_targetKDTreeDataBase(0,index),m_targetKDTreeDataBase(1,index), m_targetKDTreeDataBase(2, index));

            //是否为inf
            if (std::isinf(tmpPt(0)) || std::isinf(tmpPt(1)) || std::isinf(tmpPt(2)) ||
                std::isnan(tmpPt(0)) || std::isnan(tmpPt(1)) || std::isnan(tmpPt(2)))
            {
                continue;
            }

            Eigen::Vector3d normal;

            if (m_isGetNormals)
            {
                normal << m_targetPointCloud->points[index].normal_x, m_targetPointCloud->points[index].normal_y, m_targetPointCloud->points[index].normal_z;
            }
            else
            {
                // 对最近邻点计算法向量
                Eigen::VectorXi neighborIndices(m_search_number_normal);
                Eigen::VectorXd neighborDist2(m_search_number_normal);

                int num = m_pTargetKDTree->knn(tmpPt, neighborIndices, neighborDist2, m_search_number_normal, 0.0,
                                               Nabo::NNSearchD::SORT_RESULTS,
                                               m_r_normal);

                if (num < m_search_number_normal)
                {
                    normal(0) = normal(1) = normal(2) = std::numeric_limits<double>::infinity();
                }
                else
                {
                    std::vector<Eigen::Vector3d> nearPointsForNormal;
                    for (int ix = 0; ix < m_search_number_normal; ix++)
                    {
                        if (neighborDist2(ix) < std::numeric_limits<double>::infinity() && !std::isinf(neighborDist2(ix)))
                        {
                            nearPointsForNormal.push_back(m_targetKDTreeDataBase.col(neighborIndices(ix)));
                        }
                    }
                    normal = ComputeNormal(nearPointsForNormal);
                }
            }
            //如果对应的点没有法向量，则不要．
            if (std::isinf(normal(0)) || std::isinf(normal(1)) || std::isinf(normal(2)) ||
                std::isnan(normal(0)) || std::isnan(normal(1)) || std::isnan(normal(2)))
            {
                continue;
            }

            if (m_normal_angle_constraint)
            {
                double cos_angle = xi_normalVector.dot(normal) / (xi_normalVector.norm() * normal.norm());
                double angle = std::acos(cos_angle) * 180.0 / M_PI; // 角度转换

                // 如果角度差大于设定阈值，则跳过该点
                if (angle > m_angle_diff_threshold) {
                    continue;
                }
            }

            nearPoints.push_back(tmpPt);
            nearNormals.push_back(normal);
        }
        else
        {
            continue;
        }
    }

    //如果nearPoints小于３个，则认为没有匹配点．
    if(nearPoints.size() < 3)
    {
        return false;
    }

    double m_h_max = std::sqrt(nearDist2(nearPoints.size() - 1)) / 3;
    //根据函数进行投影．计算height，即ppt中的I(x)
    for(int i = 0; i < nearPoints.size(); ++i)
    {
        double diff_norm = (x_vec - nearPoints[i]).transpose() * (x_vec - nearPoints[i]);
        double weight = std::exp(-diff_norm / m_h_max/ m_h_max);
        double proj = weight * (x_vec - nearPoints[i]).transpose() * nearNormals[i];

        weightSum += weight;
        projSum += proj;
    }

    height = projSum / (weightSum + 1e-5);

    return true;
}


/**
 * @brief IMLSICPMatcher::ProjSourcePtToSurface
 * 此函数的目的是把source_cloud中的点云，投影到对应的surface上．
   即为每一个当前帧点云in_cloud的激光点计算匹配点和对应的法向量
   即in_cloud和out_cloud进行匹配，同时得到out_normal
   注意：本函数应该删除in_cloud内部找不到匹配值的点．
 * @param in_cloud          当前帧点云
 * @param out_cloud         当前帧的匹配点云
 * @param out_normal        匹配点云对应的法向量．
 */
void IMLSICPMatcher::ProjSourcePtToSurface(
        pcl::PointCloud<PointType>::Ptr &in_cloud,
        pcl::PointCloud<PointType>::Ptr &out_cloud,
        const std::string &timestamp, 
        const int &i)
{
    out_cloud->clear();

    std::unordered_map<Eigen::Vector3d, Eigen::Vector3d, Vector3dHash, Vector3dEqual> tensorVotingNormalsMap;

    size_t delete_no_normal = 0;
    size_t delete_too_far = 0;
    size_t delete_invalid_normal = 0;
    size_t delete_normal_constraint = 0;
    size_t delete_mls_fail = 0;
    size_t delete_nan_inf_height = 0;

    // 如果 m_isGetNormals 为 false 并且 m_useTensorVoting 为 true，使用 tensor voting
    if (!m_isGetNormals && m_useTensorVoting) {
        // 1. 将 in_cloud 转换为 DataPoints 格式
        DPD::Labels featureLabels;
        featureLabels.push_back(DPD::Label("x", 1));
        featureLabels.push_back(DPD::Label("y", 1));
        featureLabels.push_back(DPD::Label("z", 1));

        Eigen::MatrixXd features(4, in_cloud->size());
        for (size_t i = 0; i < in_cloud->size(); ++i) {
            features(0, i) = in_cloud->points[i].x;
            features(1, i) = in_cloud->points[i].y;
            features(2, i) = in_cloud->points[i].z;
            features(3, i) = 1.0; // homogeneous 坐标
        }

        DPD in_cloudDP(features, featureLabels);

        // 2. 使用 VoteForAny 计算法向量
        TensorVoting<double> tv_input(m_tensor_sigma, m_tensor_k);   // 设置合适的 sigma 和 k 值
        TensorVoting<double> tv_output(m_tensor_sigma, m_tensor_k);  // 同样为输出设置相同的 sigma 和 k

        auto [tv_output_nonZero, nonZeroIndices] = VoteForAny(tv_input, tv_output, m_targetPointCloudDP, in_cloudDP);

        // 输出nonZeroIndices.size()
        std::cout << "Number of non-zero indices from tensor voting: " << nonZeroIndices.size() << std::endl;

        // 3. 存储 tensor voting 计算的法向量
        for (size_t i = 0; i < nonZeroIndices.size(); ++i) {
            /* ---------- ATTENTION: LIBPOINTMATCHER HAS BUGS, HERE NORMAL IS ACTUALLY TANGENT IN LIBPOINTMATCHER ----------*/
            Eigen::Vector3d normal = tv_output_nonZero.tangents.col(i).cast<double>();
            Eigen::Vector3d point = features.block<3,1>(0, nonZeroIndices[i]).cast<double>();
            tensorVotingNormalsMap[point] = normal;
        }

        // save current transformed points DP
        // std::string filenameDP = m_output_dir + "dp_surface_cloud_target/" + timestamp + "_" + std::to_string(i) + ".txt";
        // saveCloudFeaturesAndDescriptors(in_cloudDP, filenameDP);
    }

    for (auto it = in_cloud->begin(); it != in_cloud->end();)
    {
        PointType &xi = *it;
        Eigen::Vector3d xi_vec(xi.x, xi.y, xi.z);
        Eigen::Vector3d xi_normalVector(xi.normal_x, xi.normal_y, xi.normal_z);

        int best_index = -1;
        double min_dist = std::numeric_limits<double>::infinity();

        // 如果使用法向量投影距离
        if (m_useProjectedDistance) {
            Eigen::Matrix3Xd target_points = m_targetKDTreeDataBase;

            // 计算每个目标点与当前点 xi_vec 的差向量
            Eigen::Matrix3Xd diff = target_points.colwise() - xi_vec;

            // 计算每个差向量与法向量 xi_normalVector 的叉乘
            Eigen::Matrix3Xd cross_product = diff.colwise().cross(xi_normalVector);

            // 计算每个叉乘结果的范数，即投影距离
            Eigen::VectorXd proj_dist = cross_product.colwise().norm();

            std::vector<std::pair<double, int>> filteredDistances;
            for (int j = 0; j < proj_dist.size(); ++j) {
                if (diff.col(j).norm() < m_r_proj && proj_dist(j) < m_r){
                    filteredDistances.emplace_back(proj_dist(j), j);
                }
            }

            // 检查是否存在满足条件的投影距离
            if (filteredDistances.size() > 0) 
            {
                // 按投影距离排序并选择最小的一个
                std::sort(filteredDistances.begin(), filteredDistances.end());
                min_dist = std::pow(filteredDistances[0].first, 2); // 取最小值并存储平方距离
                best_index = filteredDistances[0].second; 
            } 
            else 
            {
                it = in_cloud->erase(it);
                ++delete_too_far;
                continue;
            }
        } 
        else
        {
            //找到在target_cloud中的最近邻
            //包括该点和下标．
            int K = 1;
            Eigen::VectorXi indices(K);
            Eigen::VectorXd dist2(K);

            m_pTargetKDTree->knn(xi_vec, indices, dist2, K, 0.0,
                                Nabo::NNSearchD::SORT_RESULTS,
                                m_r);
            best_index = indices(0);
            min_dist = dist2(0);
        }

        if (best_index < 0 || best_index >= m_targetKDTreeDataBase.cols())
        {
            it = in_cloud->erase(it);
            ++delete_no_normal;
            continue;
        }

        //如果距离太远，则说明没有匹配点．因此可以不需要进行投影，直接去除．
        if(min_dist > m_h * m_h)
        {
            it = in_cloud->erase(it);
            ++delete_too_far;
            continue;
        }
        
        Eigen::Vector3d nearNormal;

        // Only compute normal if m_isGetNormals is false
        if (m_isGetNormals) 
        {
            nearNormal << m_targetPointCloud->points[best_index].normal_x, m_targetPointCloud->points[best_index].normal_y, m_targetPointCloud->points[best_index].normal_z;
        } 
        else if (m_useTensorVoting) 
        {
            // 使用 tensor voting 计算的法向量
            if (tensorVotingNormalsMap.find(xi_vec) != tensorVotingNormalsMap.end()) {
                nearNormal = tensorVotingNormalsMap[xi_vec];
            } else {
                it = in_cloud->erase(it); // 没有法向量，跳过该点
                ++delete_no_normal;
                continue;
            }
        } 
        else 
        {
            Eigen::VectorXi neighborIndices(m_search_number_normal);
            Eigen::VectorXd neighborDist2(m_search_number_normal);
            
            int num = m_pTargetKDTree->knn(m_targetKDTreeDataBase.col(best_index), neighborIndices, neighborDist2, m_search_number_normal, 0.0,
                                           Nabo::NNSearchD::SORT_RESULTS,
                                           m_r_normal);

            if (num < m_search_number_normal)
            {
                nearNormal(0) = nearNormal(1) = nearNormal(2) = std::numeric_limits<double>::infinity();
            }
            else
            {
                std::vector<Eigen::Vector3d> nearPoints;
                for (int ix = 0; ix < m_search_number_normal; ix++)
                {
                    if (neighborDist2(ix) < std::numeric_limits<double>::infinity() && std::isinf(neighborDist2(ix)) == false)
                    {
                        nearPoints.push_back(m_targetKDTreeDataBase.col(neighborIndices(ix)));
                    }
                }
                nearNormal = ComputeNormal(nearPoints);
            }
        }

        //如果对应的点没有法向量，也认为没有匹配点．因此直接不考虑．
        if(std::isinf(nearNormal(0))||std::isinf(nearNormal(1))||std::isinf(nearNormal(2))||
                std::isnan(nearNormal(0))||std::isnan(nearNormal(1))||std::isnan(nearNormal(2)))
        {
            it = in_cloud->erase(it);
            ++delete_invalid_normal;
            continue;
        }

        if (m_normal_angle_constraint)
        {
            double cos_angle = xi_normalVector.dot(nearNormal) / (xi_normalVector.norm() * nearNormal.norm());
            double angle = std::acos(cos_angle) * 180.0 / M_PI; // 角度转换

            // 如果角度差大于设定阈值，则跳过该点
            if (angle > m_angle_diff_threshold) {
                it = in_cloud->erase(it);
                ++delete_normal_constraint;
                continue;
            }
        }

        //进行匹配
        double height;
        if(ImplicitMLSFunction(xi,height) == false)
        {
            it = in_cloud->erase(it);
            ++delete_mls_fail;
            continue;
        }

        if(std::isnan(height))
        {
            std::cout <<"proj:this is nan, not possible"<<std::endl;
            it = in_cloud->erase(it);
            ++delete_nan_inf_height;
            continue;
        }

        if(std::isinf(height))
        {
            std::cout <<"proj:this is inf, not possible"<<std::endl;
            it = in_cloud->erase(it);
            ++delete_nan_inf_height;
            continue;
        }

        Eigen::Vector3d yi_vec;
        yi_vec = xi_vec - height * nearNormal;

        PointType yi;
        yi.x = yi_vec.x();
        yi.y = yi_vec.y();
        yi.z = yi_vec.z();

        yi.normal_x = nearNormal.x();
        yi.normal_y = nearNormal.y();
        yi.normal_z = nearNormal.z();

        out_cloud->push_back(yi);

        it++;
    }

    std::cout << "Deleted points due to no normal: " << delete_no_normal << std::endl;
    std::cout << "Deleted points due to being too far: " << delete_too_far << std::endl;
    std::cout << "Deleted points due to invalid normal: " << delete_invalid_normal << std::endl;
    if (m_normal_angle_constraint)
    {
        std::cout << "Deleted points due to normal angles larger than threshold: " << delete_normal_constraint << std::endl;
    }
    std::cout << "Deleted points due to MLS failure: " << delete_mls_fail << std::endl;
    std::cout << "Deleted points due to NaN/Inf height: " << delete_nan_inf_height << std::endl;
}

/**
 * @brief IMLSICPMatcher::ComputeNormal
 * 计算法向量
 * @param nearPoints    某个点周围的所有激光点
 * @return
 */
Eigen::Vector3d IMLSICPMatcher::ComputeNormal(std::vector<Eigen::Vector3d> &nearPoints)
{
    Eigen::Vector3d normal;

    // 计算均值 mu
    Eigen::Vector3d mu = Eigen::Vector3d(0, 0, 0);
    for(auto it = nearPoints.cbegin(); it != nearPoints.cend(); ++it)
    {
        mu += *it;
    }
    mu /= nearPoints.size();

    // 计算协方差 cov
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for(int i = 0; i != nearPoints.size(); ++i)
    {
        cov += (nearPoints[i] - mu) * (nearPoints[i] - mu).transpose();
    }
    cov /= nearPoints.size();

    // 对协方差矩阵进行特征分解, 法向量定义为最小特征值对应的特征向量
    // 计算特征值和特征向量，使用selfadjont按照对阵矩阵的算法去计算，可以让产生的vec和val按照有序排列 (由小到大)
    // 由于这里协方差矩阵为对称矩阵，所以可以使用Eigen::SelfAdjointEigenSolver，更方便
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Matrix3d evs = es.eigenvectors();
    normal = evs.col(0);


    /*
    // 或者使用EigenSolver, 形式为二维向量, 比如(2,0)(-1,0)。 不会对特征值进行排序, 产生的特征向量是单位化的
    Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
    if(es.eigenvalues()[0].real() < es.eigenvalues()[1].real())    // 或者 es.eigenvalues()(0,0).real() < es.eigenvalues(1,0).real()
        normal = es.eigenvectors().col(0).real();                  // 或者 normal = es.eigenvectors().col(0).normalized().real();
    else
        normal = es.eigenvectors().col(1).real();
    */

    // 对法向量进行归一化
    normal.normalize();

    return normal;
}


/**
 * @brief IMLSICPMatcher::Match
 * 最终使用的ICP匹配函数．暂时没有删去，实际上在laser odometry中整合了进去.
 * @param finalResult
 * @param covariance
 * @return
 */
bool IMLSICPMatcher::Match(Eigen::Matrix4d& finalResult,
                           Eigen::Matrix4d& covariance,
                           const std::string& timestamp)
{
    // 初始化估计值
    Eigen::Matrix4d result;
    result.setIdentity();
    covariance.setIdentity();

    for(int i = 0; i < m_iterations; i++)
    {
        // 根据当前估计的位姿对原始点云进行转换
        pcl::PointCloud<PointType>::Ptr in_cloud(new pcl::PointCloud<PointType>(*m_sourcePointCloud));
        for(int ix = 0; ix < m_sourcePointCloud->size();ix++)
        {
            Eigen::Vector4d origin_pt;
            origin_pt << m_sourcePointCloud->points[ix].x, m_sourcePointCloud->points[ix].y, m_sourcePointCloud->points[ix].z, 1;

            // 使用4x4变换矩阵进行转换
            Eigen::Vector4d now_pt = result * origin_pt;

            // 更新 in_cloud 中当前点的 xyz
            in_cloud->points[ix].x = now_pt(0);
            in_cloud->points[ix].y = now_pt(1);
            in_cloud->points[ix].z = now_pt(2);
        }

        std::cout << timestamp + " iter " + std::to_string(i) <<std::endl;
        
        //把sourceCloud中的点投影到targetCloud组成的平面上
        //对应的投影点即为sourceCloud的匹配点．
        //每次转换完毕之后，都需要重新计算匹配点．
        //这个函数会得到对应的匹配点．
        //本次匹配会自动删除in_cloud内部的一些找不到匹配点的点．
        //因此，这个函数出来之后，in_cloud和ref_cloud是一一匹配的．
        pcl::PointCloud<PointType>::Ptr ref_cloud(new pcl::PointCloud<PointType>);
        ProjSourcePtToSurface(in_cloud, ref_cloud, timestamp, i);

        if(in_cloud->size() < 6 || ref_cloud->size() < 6)
        {
            std::cout << "Not Enough Correspondence:" << in_cloud->size() << "," << ref_cloud->size() << std::endl;
            std::cout << "ICP Iterations Failed!!" << std::endl;
            return false;
        }
        else
        {
            std::cout << "USED POINTS FINAL: " << ref_cloud->size() << std::endl;
        }

        // add marker
        visualization_msgs::Marker normalMarker;
        normalMarker.header.frame_id = "/camera_init";
        normalMarker.ns = "normalMarker";
        normalMarker.type = visualization_msgs::Marker::LINE_LIST;
        normalMarker.action = visualization_msgs::Marker::ADD;
        normalMarker.scale.x = 0.02;
        normalMarker.pose.orientation.w = 1.0;
        ros::Time time;
        normalMarker.header.stamp = time.fromSec(std::stod(timestamp));
        visualizePCAFeatures(ref_cloud, normalMarker);
        saveMarkerToFile(normalMarker, m_output_dir + "ref_normal_markers/" + timestamp + "_" + std::to_string(i) + ".obj");

        std::vector<Eigen::Vector3d> in_cloud_vec, ref_cloud_vec;
        getXYZ(in_cloud, in_cloud_vec);
        getXYZ(ref_cloud, ref_cloud_vec);
        std::vector<Eigen::Vector3d> ref_normal;
        getNormals(ref_cloud, ref_normal);

        // 计算帧间位移．从当前的source -> target
        // choose from solver
        Eigen::Matrix4d deltaTrans;
        bool flag = SolveMotionEstimationProblemLS(in_cloud_vec,
                                                ref_cloud_vec,
                                                ref_normal,
                                                deltaTrans,
                                                timestamp,
                                                0.02);

        if(flag == false)
        {
            std::cout << "IMLS ICP Iterations Failed!!!!" << std::endl;
            return false;
        }

        // 更新位姿
        result = deltaTrans * result;

        // save matched points (ProjSourcePtToSurface return)
        std::string filenameMP = m_output_dir + "matched_points/" + timestamp + "_" + std::to_string(i) + ".txt";
        saveMatchedPointsToFile(in_cloud_vec, ref_cloud_vec, filenameMP);
        // save pose
        savePoseToFile(result, m_output_dir + "imls_iter_results.txt", timestamp);

        // 计算平移距离
        double deltadist = std::sqrt(
            std::pow(deltaTrans(0, 3), 2) + 
            std::pow(deltaTrans(1, 3), 2) + 
            std::pow(deltaTrans(2, 3), 2)
        );

        // 计算旋转角度
        // 可以通过旋转矩阵的迹来计算旋转角的绝对值
        double cos_theta = (deltaTrans.block<3,3>(0,0).trace() - 1.0) / 2.0;
        cos_theta = std::min(1.0, std::max(cos_theta, -1.0)); // 确保cos_theta在[-1, 1]之间
        double deltaAngle = std::acos(cos_theta);

        // 如果迭代条件允许，则直接退出
        if(deltadist < 0.001 && deltaAngle < (0.01/57.295))
        {
            break;
        }
    }

    finalResult = result;
    return true;
}
