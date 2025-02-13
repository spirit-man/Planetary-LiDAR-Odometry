#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <teaser/ply_io.h>
#include <teaser/registration.h>
#include <teaser/certification.h>


struct PointToPlaneCostFunctor {
    PointToPlaneCostFunctor(const Eigen::Vector3d& xj, const Eigen::Vector3d& yj, const Eigen::Vector3d& nj)
        : xj_(xj), yj_(yj), nj_(nj) {}

    template <typename T>
    bool operator()(const T* const rotation, const T* const translation, T* residual) const {
        // 将旋转和平移参数转换成Eigen对象
        Eigen::Matrix<T, 3, 1> t(translation[0], translation[1], translation[2]);
        Eigen::Quaternion<T> q(rotation[3], rotation[0], rotation[1], rotation[2]);

        // 计算Rxj + t
        Eigen::Matrix<T, 3, 1> Rxj = q * xj_.cast<T>();
        Eigen::Matrix<T, 3, 1> Rxj_plus_t = Rxj + t;

        // 计算残差
        residual[0] = nj_.cast<T>().dot(Rxj_plus_t - yj_.cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& xj, const Eigen::Vector3d& yj, const Eigen::Vector3d& nj) {
        return (new ceres::AutoDiffCostFunction<PointToPlaneCostFunctor, 1, 4, 3>(
            new PointToPlaneCostFunctor(xj, yj, nj)));
    }

private:
    const Eigen::Vector3d xj_;
    const Eigen::Vector3d yj_;
    const Eigen::Vector3d nj_;
};

struct TeaserParams {
    double noise_bound;
    bool estimate_scaling;
    int rotation_max_iterations;
    double rotation_gnc_factor;
    teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM rotation_estimation_algorithm;
    double rotation_cost_threshold;
    bool use_max_clique;
    double kcore_heuristic_threshold;
};

inline std::optional<teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM> stringToTeaserEnum(const std::string& str) {
    static const std::unordered_map<std::string, teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM> strToEnumMap = {
        {"GNC_TLS", teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS},
        {"FGR", teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR},
        {"QUATRO", teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO},
    };

    auto it = strToEnumMap.find(str);
    if (it != strToEnumMap.end()) {
        return it->second;
    } else {
        return std::nullopt;
    }
}

bool SolveMotionEstimationProblemCeres(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const int max_iterations);

bool SolveMotionEstimationProblemLS(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const std::string& timestamp,
    const double threshold);

bool SolveMotionEstimationProblemWeightedLS(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const Eigen::VectorXd& weights, // 新增权重向量
    const std::string& timestamp);

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
    const double drpm_stdev_normals);
    
bool SolveMotionEstimationProblemICP(
    const std::vector<Eigen::Vector3d>& source_cloud,
    const std::vector<Eigen::Vector3d>& ref_cloud,
    Eigen::Matrix4d& deltaTrans,
    const int max_iterations,
    const int t_epsilon,
    const int e_epsilon);

bool SolveMotionEstimationProblemTeaser(
    const std::vector<Eigen::Vector3d>& source_cloud,
    const std::vector<Eigen::Vector3d>& ref_cloud,
    Eigen::Matrix4d& deltaTrans,
    const TeaserParams& params);

bool SolveMotionEstimationProblemDRPM(
    std::vector<Eigen::Vector3d>& source_cloud,
    std::vector<Eigen::Vector3d>& ref_cloud,
    std::vector<Eigen::Vector3d>& ref_normals,
    Eigen::Matrix4d& deltaTrans,
    const Eigen::VectorXd& weights,
    const std::string& timestamp,
    const double threshold,
    const double stdev_points,
    const double stdev_normals);
