#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <teaser/ply_io.h>
#include <teaser/registration.h>
#include <teaser/certification.h>
#include "solver.h"
#include "common.h"
#include "degeneracy.h"


//已知对应点对和法向量的时候，求解相对位姿．
//source_cloud在ref_cloud下的位姿．
bool SolveMotionEstimationProblemCeres(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const int max_iterations)
{
    // 初始化旋转和平移参数
    Eigen::Quaterniond q_initial = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_initial(0.0, 0.0, 0.0);

    // 创建 Ceres 的问题实例
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);

    // 定义四元数的局部参数化
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(q_initial.coeffs().data(), 4, q_parameterization);
    problem.AddParameterBlock(t_initial.data(), 3);

    // 使用 Huber 损失函数
    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);

    for (size_t i = 0; i < source_cloud.size(); ++i) {
        // 将残差添加到问题中
        ceres::CostFunction* cost_function = PointToPlaneCostFunctor::Create(
            source_cloud[i], ref_cloud[i], ref_normals[i]);
        problem.AddResidualBlock(cost_function, loss_function, q_initial.coeffs().data(), t_initial.data());
    }

    // 配置求解器参数
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = max_iterations; // 最大迭代次数
    options.minimizer_progress_to_stdout = false;

    // 开始优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 将优化结果转换回旋转矩阵和平移向量
    Eigen::Matrix3d R = q_initial.toRotationMatrix();
    deltaTrans.setIdentity();
    deltaTrans.block<3, 3>(0, 0) = R;
    deltaTrans.block<3, 1>(0, 3) = t_initial;

    return true;
}

bool SolveMotionEstimationProblemLS(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const std::string& timestamp,
    const double threshold)
{
    // 点对数量
    size_t N = source_cloud.size();

    // 构造矩阵A和向量b
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, 6);  // N x 6 矩阵
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);     // N x 1 向量

    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d s = source_cloud[i];  // 源点s
        Eigen::Vector3d d = ref_cloud[i];     // 目标点d
        Eigen::Vector3d n = ref_normals[i];   // 法向量n

        // 填充矩阵A的第i行
        A(i, 0) = n.z() * s.y() - n.y() * s.z();  // a_i1
        A(i, 1) = n.x() * s.z() - n.z() * s.x();  // a_i2
        A(i, 2) = n.y() * s.x() - n.x() * s.y();  // a_i3
        A(i, 3) = n.x();                          // n_ix
        A(i, 4) = n.y();                          // n_iy
        A(i, 5) = n.z();                          // n_iz

        // 计算b的第i行
        b(i) = n.dot(d - s);  // 对应公式(8)中的第i个分量
    }

    // 求解Ax = b，获得x
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

    // 计算残差
    Eigen::VectorXd residuals = A * x - b;

    // for (size_t i = 0; i < residuals.size(); ++i) 
    // {
    //     saveThresholdFile(OUTPUT_DIR + "residuals/" + timestamp + ".txt", residuals[i]);
    // }

    // 对残差排序并找到2%和98%的位置
    std::vector<size_t> indices(residuals.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&residuals](size_t i1, size_t i2) {
        return std::abs(residuals(i1)) < std::abs(residuals(i2));
    });

    size_t lowerIndex = static_cast<size_t>(threshold * N);
    size_t upperIndex = static_cast<size_t>((1 - threshold) * N);

    // 根据筛选的索引重新构建A和b
    Eigen::MatrixXd A_filtered(upperIndex - lowerIndex + 1, 6);
    Eigen::VectorXd b_filtered(upperIndex - lowerIndex + 1);

    for (size_t i = lowerIndex; i <= upperIndex; ++i) {
        A_filtered.row(i - lowerIndex) = A.row(indices[i]);
        b_filtered(i - lowerIndex) = b(indices[i]);
    }

    // 使用筛选后的数据重新求解Ax = b
    x = A_filtered.colPivHouseholderQr().solve(b_filtered);


    // 提取旋转和平移参数
    Eigen::Vector3d rotation = x.segment<3>(0);   // 旋转分量
    Eigen::Vector3d translation = x.segment<3>(3); // 平移分量

    // 生成旋转矩阵
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R = Eigen::AngleAxisd(rotation.norm(), rotation.normalized()).toRotationMatrix();

    // 使用SVD对旋转矩阵进行修正，以确保其为有效的旋转矩阵
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();

    // 确保旋转矩阵的行列式为1，如果不为1则调整
    if (R.determinant() < 0)
    {
        Eigen::Matrix3d U = svd.matrixU();
        U.col(2) *= -1;
        R = U * svd.matrixV().transpose();
    }

    // 将结果写入输出的变换矩阵deltaTrans
    deltaTrans.setIdentity();
    deltaTrans.block<3, 3>(0, 0) = R;
    deltaTrans.block<3, 1>(0, 3) = translation;

    return true;
}

bool SolveMotionEstimationProblemWeightedLS(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const Eigen::VectorXd& weights, // 新增权重向量
    const std::string& timestamp)
{
    size_t N = source_cloud.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, 6);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d s = source_cloud[i];
        Eigen::Vector3d d = ref_cloud[i];
        Eigen::Vector3d n = ref_normals[i];

        A(i, 0) = n.z() * s.y() - n.y() * s.z();
        A(i, 1) = n.x() * s.z() - n.z() * s.x();
        A(i, 2) = n.y() * s.x() - n.x() * s.y();
        A(i, 3) = n.x();
        A(i, 4) = n.y();
        A(i, 5) = n.z();

        b(i) = n.dot(d - s);
    }

    // 对矩阵A和向量b应用权重
    Eigen::VectorXd sqrt_weights = weights.array().sqrt();
    A = sqrt_weights.asDiagonal() * A;
    b = sqrt_weights.asDiagonal() * b;

    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

    // 提取旋转和平移参数并生成最终变换矩阵
    Eigen::Vector3d rotation = x.segment<3>(0);
    Eigen::Vector3d translation = x.segment<3>(3);
    Eigen::Matrix3d R = Eigen::AngleAxisd(rotation.norm(), rotation.normalized()).toRotationMatrix();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3d U = svd.matrixU();
        U.col(2) *= -1;
        R = U * svd.matrixV().transpose();
    }

    deltaTrans.setIdentity();
    deltaTrans.block<3, 3>(0, 0) = R;
    deltaTrans.block<3, 1>(0, 3) = translation;

    return true;
}

bool SolveMotionEstimationProblemRANSAC(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const std::string& timestamp,
    const int max_iterations,
    const double distance_threshold,
    const double min_inliers_percentage,
    const double huber_threshold,
    const std::string final_solve_method,
    const double ls_threshold,
    const double drpm_threshold,
    const double drpm_stdev_points,
    const double drpm_stdev_normals)
{
    const int min_inliers = static_cast<int>(min_inliers_percentage * source_cloud.size()); // 最小内点数量

    int best_inliers_count = 0;
    Eigen::Matrix4d best_transformation = Eigen::Matrix4d::Identity();

    // RANSAC 迭代
    for (int iter = 0; iter < max_iterations; ++iter) {
        // FPS sampling
        std::vector<int> indices;
        farthestPointSampling(source_cloud, 3, indices);

        // 估算变换矩阵
        Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
        Eigen::MatrixXd A(3, 6);  // 3 个点对
        Eigen::VectorXd b(3);     // 3 个点对的 b 值

        // 根据这三个点对填充 A 和 b
        for (size_t i = 0; i < 3; ++i) {
            Eigen::Vector3d s = source_cloud[indices[i]];  // 源点 s
            Eigen::Vector3d d = ref_cloud[indices[i]];     // 目标点 d
            Eigen::Vector3d n = ref_normals[indices[i]];   // 法向量 n

            // 填充矩阵 A 的第 i 行
            A(i, 0) = n.z() * s.y() - n.y() * s.z();  // a_i1
            A(i, 1) = n.x() * s.z() - n.z() * s.x();  // a_i2
            A(i, 2) = n.y() * s.x() - n.x() * s.y();  // a_i3
            A(i, 3) = n.x();                          // n_ix
            A(i, 4) = n.y();                          // n_iy
            A(i, 5) = n.z();                          // n_iz

            // 计算 b 的第 i 行
            b(i) = n.dot(d - s);  // 对应公式中的第 i 个分量
        }

        // 求解 Ax = b，获得 x
        Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

        // 提取旋转和平移参数
        Eigen::Vector3d rotation = x.segment<3>(0);   // 旋转分量
        Eigen::Vector3d translation = x.segment<3>(3); // 平移分量

        // 生成旋转矩阵
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        R = Eigen::AngleAxisd(rotation.norm(), rotation.normalized()).toRotationMatrix();

        // 使用 SVD 对旋转矩阵进行修正，以确保其为有效的旋转矩阵
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();

        // 确保旋转矩阵的行列式为 1，如果不为 1 则调整
        if (R.determinant() < 0)
        {
            Eigen::Matrix3d U = svd.matrixU();
            U.col(2) *= -1;
            R = U * svd.matrixV().transpose();
        }

        // 构建变换矩阵
        transformation.setIdentity();
        transformation.block<3, 3>(0, 0) = R;
        transformation.block<3, 1>(0, 3) = translation;

        int inliers_count = 0;
        for (size_t i = 0; i < source_cloud.size(); ++i) {
            Eigen::Vector3d s = source_cloud[i];
            Eigen::Vector3d d = ref_cloud[i];
            Eigen::Vector3d n = ref_normals[i];

            // 使用当前估算的变换矩阵
            Eigen::Vector3d transformed_point = transformation.block<3, 3>(0, 0) * s + transformation.block<3, 1>(0, 3);
            double distance = std::abs((transformed_point - d).dot(n));

            // 如果点对在距离阈值内，则认为是内点
            if (distance < distance_threshold) {
                ++inliers_count;
            }
        }

        // 更新最佳变换矩阵
        if (inliers_count > best_inliers_count) {
            best_inliers_count = inliers_count;
            best_transformation = transformation;
        }

        // 如果内点数量超过最小内点数，提前结束
        if (best_inliers_count > min_inliers) {
            break;
        }
    }

    // 使用最佳变换矩阵进行优化
    std::vector<Eigen::Vector3d> inlier_source_cloud;
    std::vector<Eigen::Vector3d> inlier_ref_cloud;
    std::vector<Eigen::Vector3d> inlier_ref_normals;
    std::vector<double> weights_vec;
    // double huber_threshold2 = 0.873 * distance_threshold; // 1 / 1.145
    double huber_threshold2 = huber_threshold * distance_threshold; // 卡方分布95%的值, 0.648

    for (size_t i = 0; i < source_cloud.size(); ++i) {
        Eigen::Vector3d s = source_cloud[i];
        Eigen::Vector3d d = ref_cloud[i];
        Eigen::Vector3d n = ref_normals[i];

        // 使用最佳变换矩阵计算残差
        Eigen::Vector3d transformed_point = best_transformation.block<3, 3>(0, 0) * s + best_transformation.block<3, 1>(0, 3);
        double distance = std::abs((transformed_point - d).dot(n));

        // 如果残差小于阈值，则是内点
        if (distance < distance_threshold) {
            inlier_source_cloud.push_back(s);
            inlier_ref_cloud.push_back(d);
            inlier_ref_normals.push_back(n);
            double abs_residual = std::exp(-std::abs(distance));
            if (std::sqrt(abs_residual) < huber_threshold2) {
                weights_vec.push_back(abs_residual);
            } else {
                weights_vec.push_back(2 * huber_threshold2 * std::sqrt(abs_residual) - huber_threshold2 * huber_threshold2);
            }
        }
    }

    Eigen::VectorXd weights = Eigen::Map<Eigen::VectorXd>(weights_vec.data(), weights_vec.size());

    double weight_sum = weights.sum();
    if (weight_sum > 0) {
        weights /= weight_sum;
    }

    std::cout << "Inlier source_cloud size: " + std::to_string(inlier_source_cloud.size()) << std::endl;

    if (final_solve_method == "LS")
    {
        return SolveMotionEstimationProblemLS(inlier_source_cloud, inlier_ref_cloud, inlier_ref_normals, deltaTrans, timestamp, ls_threshold);
    }
    else if (final_solve_method == "Weighted LS")
    {
        return SolveMotionEstimationProblemWeightedLS(inlier_source_cloud, inlier_ref_cloud, inlier_ref_normals, deltaTrans, weights, timestamp);
    }
    else if (final_solve_method == "DRPM")
    {
        return SolveMotionEstimationProblemDRPM(inlier_source_cloud, inlier_ref_cloud, inlier_ref_normals, deltaTrans, weights, timestamp, drpm_threshold, drpm_stdev_points, drpm_stdev_normals);
    }
    else
    {
        std::cout << "Invalid FINAL_SOLVE_METHOD in RANSAC!" << std::endl;
        return false;
    }
}

bool SolveMotionEstimationProblemICP(
    const std::vector<Eigen::Vector3d>& source_cloud,
    const std::vector<Eigen::Vector3d>& ref_cloud,
    Eigen::Matrix4d& deltaTrans,
    const int max_iterations,
    const int t_epsilon,
    const int e_epsilon)
{
    // 检查输入点云大小是否一致
    if (source_cloud.size() != ref_cloud.size() || source_cloud.empty()) {
        return false;
    }

    // 定义PCL点云类型
    using PointT = pcl::PointXYZ;
    pcl::PointCloud<PointT>::Ptr src_cloud(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr tgt_cloud(new pcl::PointCloud<PointT>());

    // 将Eigen点转换为PCL点云
    for (const auto& pt : source_cloud) {
        src_cloud->points.emplace_back(pt.x(), pt.y(), pt.z());
    }
    for (const auto& pt : ref_cloud) {
        tgt_cloud->points.emplace_back(pt.x(), pt.y(), pt.z());
    }

    // 设置ICP对象
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(src_cloud);
    icp.setInputTarget(tgt_cloud);
    icp.setMaximumIterations(max_iterations);
    icp.setTransformationEpsilon(t_epsilon);
    icp.setEuclideanFitnessEpsilon(e_epsilon);

    // 执行配准
    pcl::PointCloud<PointT> Final;
    icp.align(Final);

    // 检查是否收敛
    if (icp.hasConverged()) {
        // 获取变换矩阵
        Eigen::Matrix4f transformation = icp.getFinalTransformation();
        deltaTrans = transformation.cast<double>();
        return true;
    } else {
        return false;
    }
}

bool SolveMotionEstimationProblemTeaser(
    const std::vector<Eigen::Vector3d>& source_cloud,
    const std::vector<Eigen::Vector3d>& ref_cloud,
    Eigen::Matrix4d& deltaTrans,
    const TeaserParams& params)
{
    if (source_cloud.size() != ref_cloud.size() || source_cloud.empty()) {
        return false;
    }

    // 将 std::vector 转换为 Eigen 矩阵
    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, source_cloud.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, ref_cloud.size());

    for (size_t i = 0; i < source_cloud.size(); ++i) {
        src.col(i) = source_cloud[i];
        tgt.col(i) = ref_cloud[i];
    }

    // 设置 TEASER++ 参数
    teaser::RobustRegistrationSolver::Params teaser_params;
    teaser_params.noise_bound = params.noise_bound;
    teaser_params.estimate_scaling = params.estimate_scaling;
    teaser_params.rotation_max_iterations = params.rotation_max_iterations;
    teaser_params.rotation_gnc_factor = params.rotation_gnc_factor;
    teaser_params.rotation_estimation_algorithm = params.rotation_estimation_algorithm;
    teaser_params.rotation_cost_threshold = params.rotation_cost_threshold;
    teaser_params.use_max_clique = params.use_max_clique;
    teaser_params.kcore_heuristic_threshold = params.kcore_heuristic_threshold;

    // 初始化 TEASER++ 求解器
    teaser::RobustRegistrationSolver solver(teaser_params);

    // 执行配准
    solver.solve(src, tgt);
    auto solution = solver.getSolution();

    if (!solution.valid) {
        return false;
    }

    // 构建变换矩阵
    deltaTrans = Eigen::Matrix4d::Identity();
    deltaTrans.block<3, 3>(0, 0) = solution.rotation;
    deltaTrans.block<3, 1>(0, 3) = solution.translation;

    return true;
}


std::vector<Eigen::Matrix3d> GetIsotropicCovariances(
    const size_t& N, 
    const double stdev) 
{
    std::vector<Eigen::Matrix3d> covariances;
    covariances.reserve(N);
    
    for (size_t i = 0; i < N; i++) {
        covariances.push_back(Eigen::Matrix3d::Identity() * std::pow(stdev, 2));
    }
    return covariances;
}

bool SolveMotionEstimationProblemDRPM(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const Eigen::VectorXd& weights,
    const std::string& timestamp,
    const double threshold,
    const double stdev_points,
    const double stdev_normals)
{
    size_t N = source_cloud.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, 6);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

    // 1. 构建A矩阵和b向量
    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d s = source_cloud[i];
        Eigen::Vector3d d = ref_cloud[i];
        Eigen::Vector3d n = ref_normals[i];

        A(i, 0) = n.z() * s.y() - n.y() * s.z();
        A(i, 1) = n.x() * s.z() - n.z() * s.x();
        A(i, 2) = n.y() * s.x() - n.x() * s.y();
        A(i, 3) = n.x();
        A(i, 4) = n.y();
        A(i, 5) = n.z();

        b(i) = n.dot(d - s);
    }

    // 应用权重
    Eigen::VectorXd sqrt_weights = weights.array().sqrt();
    Eigen::MatrixXd weighted_A = sqrt_weights.asDiagonal() * A;
    Eigen::VectorXd weighted_b = sqrt_weights.asDiagonal() * b;

    // 2. 计算加权Hessian矩阵
    const auto normal_covariances = GetIsotropicCovariances(ref_normals.size(), stdev_normals);
    Eigen::Matrix<double, 6, 6> H = weighted_A.transpose() * weighted_A;

    // 3. 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(H);
    const auto eigenvectors = eigensolver.eigenvectors();
    const auto eigenvalues = eigensolver.eigenvalues();

    // 4. 计算噪声估计
    Eigen::Matrix<double, 6, 6> noise_mean;
    Eigen::Matrix<double, 6, 1> noise_variance;
    const double snr_factor = 10.0;

    std::tie(noise_mean, noise_variance) = 
        degeneracy::ComputeNoiseEstimate<double, double>(
            source_cloud, ref_normals, weights, 
            normal_covariances, eigenvectors, stdev_points);

    // 5. 计算退化概率
    Eigen::Matrix<double, 6, 1> non_degeneracy_probabilities = 
        degeneracy::ComputeSignalToNoiseProbabilities<double>(
            H, noise_mean, noise_variance, eigenvectors, snr_factor);

    std::cout << "The non-degeneracy probabilities are: " << std::endl;
    std::cout << non_degeneracy_probabilities.transpose() << std::endl;

    std::cout << "For the eigenvectors of the Hessian: " << std::endl;
    std::cout << eigenvectors << std::endl;

    // 6. 求解
    Eigen::VectorXd x;
    if (non_degeneracy_probabilities.minCoeff() < threshold) {
        // 使用修改后的DRPM求解
        x = degeneracy::SolveWithSnrProbabilities<double>(
            eigenvectors, 
            eigenvalues, 
            weighted_A.transpose() * weighted_b,  // 使用加权的右侧向量
            non_degeneracy_probabilities);
    } else {
        // 使用加权最小二乘求解
        x = weighted_A.colPivHouseholderQr().solve(weighted_b);
    }

    // 7. 构建变换矩阵
    Eigen::Vector3d rotation = x.segment<3>(0);
    Eigen::Vector3d translation = x.segment<3>(3);

    // 构建旋转矩阵
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R = Eigen::AngleAxisd(rotation.norm(), rotation.normalized()).toRotationMatrix();

    // SVD修正
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();

    if (R.determinant() < 0) {
        Eigen::Matrix3d U = svd.matrixU();
        U.col(2) *= -1;
        R = U * svd.matrixV().transpose();
    }

    // 构建最终变换矩阵
    deltaTrans.setIdentity();
    deltaTrans.block<3, 3>(0, 0) = R;
    deltaTrans.block<3, 1>(0, 3) = translation;

    return true;
}

