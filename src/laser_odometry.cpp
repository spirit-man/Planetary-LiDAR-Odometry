#include <mutex>
#include <queue>
#include <deque>
#include <fstream>
#include <iomanip>
#include <thread>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <teaser/ply_io.h>
#include <teaser/registration.h>
#include <teaser/certification.h>
#include <nabo/nabo.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include "imls_icp.h"
#include "tic_toc.h"
#include "common.h"
#include "saver.h"
#include "solver.h"


#define DISTORTION 0
constexpr double SCAN_PERIOD = 0.1;

std::mutex mBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cloudBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> flatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> lessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cloudBufDP;
std::queue<sensor_msgs::PointCloud2ConstPtr> flatBufDP;

ros::Publisher pubOdoPath;
nav_msgs::Path odoPath;

ros::Subscriber subLaserCloud;
ros::Subscriber subLaserFlat;
ros::Subscriber subLaserLessFlat;
ros::Subscriber subLaserCloudDP;
ros::Subscriber subLaserFlatDP;

Eigen::Matrix4d prevLaserPose = Eigen::Matrix4d::Identity();
Eigen::Quaterniond q_global_curr(1, 0, 0, 0);
Eigen::Vector3d t_global_curr(0, 0, 0);
Eigen::Quaterniond q_last_curr(1, 0, 0, 0);
Eigen::Vector3d t_last_curr(0, 0, 0);

std::deque<pcl::PointCloud<PointType>::Ptr> cloudQueue;
pcl::PointCloud<PointType>::Ptr accumulatedTargetCloud(new pcl::PointCloud<PointType>());
int frameCount = 0;
DP prevFilteredLaserCloudDP;

std::string pose_file;


void TransformToStart(pcl::PointCloud<PointType>::Ptr& cloud, 
                      const Eigen::Quaterniond& q_last_curr, 
                      const Eigen::Vector3d& t_last_curr,
                      bool transform_normal) 
{
    Eigen::Matrix3d rotation_matrix = q_last_curr.toRotationMatrix();

    for (auto& point : cloud->points) {
        // 处理点的位置
        Eigen::Vector3d p(point.x, point.y, point.z);
        p = rotation_matrix * p + t_last_curr;
        point.x = p.x();
        point.y = p.y();
        point.z = p.z();

        if (transform_normal)
        {
            // 处理点的法向量
            Eigen::Vector3d n(point.normal_x, point.normal_y, point.normal_z);
            n = rotation_matrix * n;
            point.normal_x = n.x();
            point.normal_y = n.y();
            point.normal_z = n.z();
        }
    }
}

void TransformToEnd(pcl::PointCloud<PointType>::Ptr& cloud, 
                    const Eigen::Quaterniond& q_last_curr, 
                    const Eigen::Vector3d& t_last_curr,
                    bool transform_normal) 
{
    Eigen::Matrix3d rotation_matrix = q_last_curr.inverse().toRotationMatrix();

    for (auto& point : cloud->points) {
        // 处理点的位置
        Eigen::Vector3d p(point.x, point.y, point.z);
        p = rotation_matrix * (p - t_last_curr);
        point.x = p.x();
        point.y = p.y();
        point.z = p.z();

        if (transform_normal)
        {
            // 处理点的法向量
            Eigen::Vector3d n(point.normal_x, point.normal_y, point.normal_z);
            n = rotation_matrix * n;
            point.normal_x = n.x();
            point.normal_y = n.y();
            point.normal_z = n.z();
        }
    }
}

void accumulateTargetCloud(pcl::PointCloud<PointType>::Ptr& newCloud, const size_t max_queue_size, bool transform_normal) 
{
    // 首先将新点云转换到当前帧结束时刻的坐标系
    // TransformToEnd(newCloud, q_last_curr, t_last_curr, transform_normal);

    // 将前面的累积点云转换到当前帧结束时刻的坐标系
    // for (auto& cloud : cloudQueue) {
    //     TransformToEnd(cloud, q_last_curr, t_last_curr, transform_normal);
    // }

    cloudQueue.push_back(newCloud);

    if (cloudQueue.size() > max_queue_size) {
        cloudQueue.pop_front();
    }

    accumulatedTargetCloud->clear();
    for (const auto& cloud : cloudQueue) {
        *accumulatedTargetCloud += *cloud;
    }
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{
    mBuf.lock();
    cloudBuf.push(cloudMsg);
    mBuf.unlock();
}

void laserFlatHandler(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{
    mBuf.lock();
    flatBuf.push(cloudMsg);
    mBuf.unlock();
}

void laserLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{
    mBuf.lock();
    lessFlatBuf.push(cloudMsg);
    mBuf.unlock();
}

void laserCloudHandlerDP(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{
    mBuf.lock();
    cloudBufDP.push(cloudMsg);
    mBuf.unlock();
}

void laserFlatHandlerDP(const sensor_msgs::PointCloud2ConstPtr &cloudMsg)
{
    mBuf.lock();
    flatBufDP.push(cloudMsg);
    mBuf.unlock();
}

bool solveMotionEstimationProblem(
	const std::string& solve_method,
	std::vector<Eigen::Vector3d>& in_cloud_vec,
	std::vector<Eigen::Vector3d>& ref_cloud_vec,
	std::vector<Eigen::Vector3d>& ref_normal,
	Eigen::Matrix4d& deltaTrans,
	std::string& timestamp)
{
    bool flag = false;
	if(solve_method=="Ceres"){
        int max_iterations = config["laser_odometry"]["solve_method"]["Ceres"]["max_iterations"];
        flag = SolveMotionEstimationProblemCeres(in_cloud_vec,
                                            ref_cloud_vec,
                                            ref_normal,
                                            deltaTrans,
                                            max_iterations);
    }
    else if (solve_method=="LS"){
        double threshold = config["laser_odometry"]["solve_method"]["LS"]["threshold"];
        flag = SolveMotionEstimationProblemLS(in_cloud_vec,
                                            ref_cloud_vec,
                                            ref_normal,
                                            deltaTrans,
                                            timestamp,
                                            threshold);
    }
    else if (solve_method=="RANSAC"){
        int max_iterations = config["laser_odometry"]["solve_method"]["RANSAC"]["max_iterations"];
        double distance_threshold = config["laser_odometry"]["solve_method"]["RANSAC"]["distance_threshold"];
        double min_inliers_percentage = config["laser_odometry"]["solve_method"]["RANSAC"]["min_inliers_percentage"];
        double huber_threshold = config["laser_odometry"]["solve_method"]["RANSAC"]["huber_threshold"];
        std::string final_solve_method = config["laser_odometry"]["solve_method"]["RANSAC"]["final_solve_method"];
        double ls_threshold = config["laser_odometry"]["solve_method"]["RANSAC"]["LS_threshold"];
        double drpm_threshold = config["laser_odometry"]["solve_method"]["RANSAC"]["DRPM_threshold"];
        double drpm_stdev_points = config["laser_odometry"]["solve_method"]["RANSAC"]["DRPM_stdev_points"];
        double drpm_stdev_normals = config["laser_odometry"]["solve_method"]["RANSAC"]["DRPM_stdev_normals"];
        flag = SolveMotionEstimationProblemRANSAC(in_cloud_vec,
                                            ref_cloud_vec,
                                            ref_normal,
                                            deltaTrans,
                                            timestamp,
                                            max_iterations,
                                            distance_threshold,
                                            min_inliers_percentage,
                                            huber_threshold,
                                            final_solve_method,
                                            ls_threshold,
                                            drpm_threshold,
                                            drpm_stdev_points,
                                            drpm_stdev_normals);
    }
    else if (solve_method=="ICP"){
        int max_iterations = config["laser_odometry"]["solve_method"]["ICP"]["max_iterations"];
        float t_epsilon = config["laser_odometry"]["solve_method"]["ICP"]["t_epsilon"];
        float e_epsilon = config["laser_odometry"]["solve_method"]["ICP"]["e_epsilon"];
        flag = SolveMotionEstimationProblemICP(in_cloud_vec,
                                            ref_cloud_vec,
                                            deltaTrans,
                                            max_iterations,
                                            t_epsilon,
                                            e_epsilon);
    }
    else if (solve_method=="Teaser"){
        double noise_bound = config["laser_odometry"]["solve_method"]["Teaser"]["noise_bound"];
        bool estimate_scaling = config["laser_odometry"]["solve_method"]["Teaser"]["estimate_scaling"];
        int rotation_max_iterations = config["laser_odometry"]["solve_method"]["Teaser"]["rotation_max_iterations"];
        double rotation_gnc_factor = config["laser_odometry"]["solve_method"]["Teaser"]["rotation_gnc_factor"];
        double rotation_cost_threshold = config["laser_odometry"]["solve_method"]["Teaser"]["rotation_cost_threshold"];
        bool use_max_clique = config["laser_odometry"]["solve_method"]["Teaser"]["use_max_clique"];
        double kcore_heuristic_threshold = config["laser_odometry"]["solve_method"]["Teaser"]["kcore_heuristic_threshold"];

        auto algorithm_opt = stringToTeaserEnum(config["laser_odometry"]["solve_method"]["Teaser"]["rotation_estimation_algorithm"]);
        
        if (algorithm_opt.has_value()) 
        {
            teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM rotation_estimation_algorithm = algorithm_opt.value();
            TeaserParams params;
            params.noise_bound = noise_bound;
            params.estimate_scaling = estimate_scaling;
            params.rotation_max_iterations = rotation_max_iterations;
            params.rotation_gnc_factor = rotation_gnc_factor;
            params.rotation_cost_threshold = rotation_cost_threshold;
            params.use_max_clique = use_max_clique;
            params.kcore_heuristic_threshold = kcore_heuristic_threshold;
            params.rotation_estimation_algorithm = rotation_estimation_algorithm;
            
            flag = SolveMotionEstimationProblemTeaser(in_cloud_vec,
                                                    ref_cloud_vec,
                                                    deltaTrans,
                                                    params);
        } 
        else 
        {
            std::cerr << "Invalid ROTATION_ESTIMATION_ALGORITHM! " << std::endl;
        }
    }
    else
    {
        std::cerr << "Invalid SOLVE_METHOD!" << std::endl;
    }

    return flag;
}

void plane_ICP_proj(
    Eigen::MatrixXd& target_KDtree_database,
    Nabo::NNSearchD* p_target_KDtree,
    pcl::PointCloud<PointType>::Ptr &in_cloud,
    pcl::PointCloud<PointType>::Ptr &ref_cloud)
{
    if(p_target_KDtree == NULL)
    {
        target_KDtree_database.resize(3,accumulatedTargetCloud->size());
        for(int i = 0; i < accumulatedTargetCloud->size();i++)
        {
            target_KDtree_database(0,i) = (*accumulatedTargetCloud)[i].x;
            target_KDtree_database(1,i) = (*accumulatedTargetCloud)[i].y;
            target_KDtree_database(2,i) = (*accumulatedTargetCloud)[i].z;
        }
        p_target_KDtree = Nabo::NNSearchD::createKDTreeLinearHeap(target_KDtree_database);
    }

    double r = config["laser_odometry"]["matching_method"]["plane_ICP"]["r"];
    bool use_projected_distance = config["laser_odometry"]["matching_method"]["plane_ICP"]["use_projected_distance"]["enabled"];
    double r_proj = config["laser_odometry"]["matching_method"]["plane_ICP"]["use_projected_distance"]["r_proj"];
    bool normal_angle_constraint = config["laser_odometry"]["matching_method"]["plane_ICP"]["normal_angle_constraint"]["enabled"];
    double angle_diff_threshold = config["laser_odometry"]["matching_method"]["plane_ICP"]["normal_angle_constraint"]["angle_diff_threshold"];

    size_t delete_no_normal = 0;
    size_t delete_too_far = 0;
    size_t delete_invalid_normal = 0;
    size_t delete_normal_constraint = 0;

    for (auto it = in_cloud->begin(); it != in_cloud->end();)
    {
        PointType &xi = *it;
        Eigen::Vector3d xi_vec(xi.x, xi.y, xi.z);
        Eigen::Vector3d xi_normalVector(xi.normal_x, xi.normal_y, xi.normal_z);

        int best_index = -1;
        double min_dist = std::numeric_limits<double>::infinity();

        if (use_projected_distance) {
            Eigen::MatrixXd diff = target_KDtree_database.colwise() - xi_vec;
            Eigen::MatrixXd cross_product = diff.colwise().cross(xi_normalVector);
            Eigen::VectorXd proj_dist = cross_product.colwise().norm();

            std::vector<std::pair<double, int>> filteredDistances;
            for (int j = 0; j < proj_dist.size(); ++j) {
                if (diff.col(j).norm() < r*r && proj_dist(j) < r_proj){
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
            int K = 1;
            Eigen::VectorXi indices(K);
            Eigen::VectorXd dist2(K);

            p_target_KDtree->knn(xi_vec, indices, dist2, K, 0.0,
                                Nabo::NNSearchD::SORT_RESULTS,
                                r);
            best_index = indices(0);
            min_dist = dist2(0);
        }

        if (best_index < 0 || best_index >= target_KDtree_database.cols())
        {
            it = in_cloud->erase(it);
            ++delete_no_normal;
            continue;
        }
        
        Eigen::Vector3d nearPoint, nearNormal;
        nearPoint << accumulatedTargetCloud->points[best_index].x, accumulatedTargetCloud->points[best_index].y, accumulatedTargetCloud->points[best_index].z;
        nearNormal << accumulatedTargetCloud->points[best_index].normal_x, accumulatedTargetCloud->points[best_index].normal_y, accumulatedTargetCloud->points[best_index].normal_z;
        
        if(std::isinf(nearNormal(0))||std::isinf(nearNormal(1))||std::isinf(nearNormal(2))||
                        std::isnan(nearNormal(0))||std::isnan(nearNormal(1))||std::isnan(nearNormal(2)))
        {
            it = in_cloud->erase(it);
            ++delete_invalid_normal;
            continue;
        }

        if (normal_angle_constraint)
        {
            double cos_angle = xi_normalVector.dot(nearNormal) / (xi_normalVector.norm() * nearNormal.norm());
            double angle = std::acos(cos_angle) * 180.0 / M_PI; // 角度转换

            // 如果角度差大于设定阈值，则跳过该点
            if (angle > angle_diff_threshold) {
                it = in_cloud->erase(it);
                ++delete_normal_constraint;
                continue;
            }
        }

        Eigen::Vector3d vec = xi_vec - nearPoint;
        double projection_distance = vec.dot(nearNormal);

        Eigen::Vector3d yi_vec = xi_vec - projection_distance * nearNormal;

        PointType yi;
        yi.x = yi_vec.x();
        yi.y = yi_vec.y();
        yi.z = yi_vec.z();

        yi.normal_x = nearNormal.x();
        yi.normal_y = nearNormal.y();
        yi.normal_z = nearNormal.z();

        ref_cloud->push_back(yi);

        it++;
    }

    std::cout << "Deleted points due to no normal: " << delete_no_normal << std::endl;
    std::cout << "Deleted points due to being too far: " << delete_too_far << std::endl;
    std::cout << "Deleted points due to invalid normal: " << delete_invalid_normal << std::endl;
    if (normal_angle_constraint)
    {
        std::cout << "Deleted points due to normal angles larger than threshold: " << delete_normal_constraint << std::endl;
    }
}


void processData() 
{
    while (ros::ok()) {
        TicToc t_whole;
        TicToc t_step;

        // if (!cloudBuf.empty() && !flatBuf.empty() && !cloudBufDP.empty() && !flatBufDP.empty()) {
        if (!cloudBuf.empty() && !flatBuf.empty() &&
            (config["scan_registration"]["presample_method"]["method"]=="tensor_voting" ? 
            (!cloudBufDP.empty() && !flatBufDP.empty()) : true)) 
        {
            
            /* -------------------- 1. get pointclouds, undistortion -------------------- */
    
            t_step.tic();

            mBuf.lock();
            pcl::PointCloud<PointType>::Ptr filteredLaserCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr flatCloud(new pcl::PointCloud<PointType>());

            pcl::fromROSMsg(*cloudBuf.front(), *filteredLaserCloud);
            ros::Time time = cloudBuf.front()->header.stamp;
            std::string timestamp = std::to_string(time.toSec());

            cloudBuf.pop();

            pcl::fromROSMsg(*flatBuf.front(), *flatCloud);
            flatBuf.pop();

            DP filteredLaserCloudDP, flatCloudDP;
            if (config["scan_registration"]["presample_method"]["method"]=="tensor_voting")
            {
                // 将 ROS 点云消息转换为 DataPoints
                filteredLaserCloudDP = rosMsgToLibPointMatcherCloud(*cloudBufDP.front());
                cloudBufDP.pop();
                flatCloudDP = rosMsgToLibPointMatcherCloud(*flatBufDP.front());
                flatBufDP.pop();
            }

            mBuf.unlock();

            // 将当前帧的flatCloud转换到上一帧结束时的坐标系
            bool transform_normal = config["laser_odometry"]["transform_normal"];
            // TransformToStart(flatCloud, q_last_curr, t_last_curr, transform_normal);

            std::string timesFile = OUTPUT_DIR + "laser_odometry_times.txt";

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

            t_step.tocAndLog("1. Preprocessing", timesFile); 


            if (frameCount != 0) {

                /* -------------------- 2. matching in flat points --------------------*/

                t_step.tic();

                Eigen::Matrix4d rPose;
                rPose.setIdentity();
                
                std::string matching_method = config["laser_odometry"]["matching_method"]["method"];
                int iterations = config["laser_odometry"]["solve_method"]["iterations"];
                IMLSICPMatcher matcher;

                if (matching_method == "IMLS")
                {
                    double h = config["laser_odometry"]["matching_method"]["IMLS"]["h"];
                    double r = config["laser_odometry"]["matching_method"]["IMLS"]["r"];
                    bool use_tensor_voting = config["laser_odometry"]["matching_method"]["IMLS"]["use_tensor_voting"]["enabled"];
                    bool is_get_normals = config["laser_odometry"]["matching_method"]["IMLS"]["get_normals"]["enabled"];
                    bool use_projected_distance = config["laser_odometry"]["matching_method"]["IMLS"]["use_projected_distance"]["enabled"];
                    int tensor_k = config["laser_odometry"]["matching_method"]["IMLS"]["use_tensor_voting"]["k"];
                    double tensor_sigma = config["laser_odometry"]["matching_method"]["IMLS"]["use_tensor_voting"]["sigma"];
                    double tensor_distance_threshold = config["laser_odometry"]["matching_method"]["IMLS"]["use_tensor_voting"]["distance_threshold"];
                    double r_normal = config["laser_odometry"]["matching_method"]["IMLS"]["get_normals"]["r_normal"];
                    int search_number_normal = config["laser_odometry"]["matching_method"]["IMLS"]["get_normals"]["search_number_normal"];
                    double r_proj = config["laser_odometry"]["matching_method"]["IMLS"]["use_projected_distance"]["r_proj"];
                    int search_number = config["laser_odometry"]["matching_method"]["IMLS"]["IMLS function"]["search_number"];
                    bool normal_angle_constraint = config["laser_odometry"]["matching_method"]["IMLS"]["normal_angle_constraint"]["enabled"];
                    double angle_diff_threshold = config["laser_odometry"]["matching_method"]["IMLS"]["normal_angle_constraint"]["angle_diff_threshold"];


                    matcher.setSourcePointCloud(flatCloud);
                    matcher.setTargetPointCloud(accumulatedTargetCloud);
                    if (prevFilteredLaserCloudDP.getNbPoints() > 0) {
                        matcher.setTargetPointCloudDP(prevFilteredLaserCloudDP);
                    }
                    matcher.setParameters(iterations, h, r, r_normal, r_proj,
                        use_tensor_voting, is_get_normals, use_projected_distance,
                        tensor_k, tensor_sigma, tensor_distance_threshold,
                        search_number_normal, search_number, normal_angle_constraint,
                        angle_diff_threshold, OUTPUT_DIR);
                }

                Eigen::MatrixXd target_KDtree_database;
                Nabo::NNSearchD* p_target_KDtree;

                for(int i = 0; i < iterations; i++)
                {
                    // 根据当前估计的位姿对原始点云进行转换
                    pcl::PointCloud<PointType>::Ptr in_cloud(new pcl::PointCloud<PointType>(*flatCloud));
                    for(int ix = 0; ix < flatCloud->size();ix++)
                    {
                        Eigen::Vector4d origin_pt;
                        origin_pt << flatCloud->points[ix].x, flatCloud->points[ix].y, flatCloud->points[ix].z, 1;

                        // 使用4x4变换矩阵进行转换
                        Eigen::Vector4d now_pt = rPose * origin_pt;

                        // 更新 in_cloud 中当前点的 xyz
                        in_cloud->points[ix].x = now_pt(0);
                        in_cloud->points[ix].y = now_pt(1);
                        in_cloud->points[ix].z = now_pt(2);

                        if (transform_normal)
                        {
                            Eigen::Vector3d n(flatCloud->points[ix].normal_x, flatCloud->points[ix].normal_y, flatCloud->points[ix].normal_z);
                            n = rPose.block<3,3>(0,0) * n;
                            in_cloud->points[ix].normal_x = n.x();
                            in_cloud->points[ix].normal_y = n.y();
                            in_cloud->points[ix].normal_z = n.z();
                        }
                    }

                    std::cout << timestamp + " iter " + std::to_string(i) << std::endl;
                    
                    pcl::PointCloud<PointType>::Ptr ref_cloud(new pcl::PointCloud<PointType>);


                    // Matching
                    if (matching_method == "IMLS")
                    {
                        matcher.ProjSourcePtToSurface(in_cloud, ref_cloud, timestamp, i);
                    }
                    else if (matching_method == "plane_ICP")
                    {
                        plane_ICP_proj(target_KDtree_database, p_target_KDtree, in_cloud, ref_cloud);
                    }
                    else
                    {
                        std::cerr << "Invalid MATCHING_METHOD!" << std::endl;
                    }

                    int correspond_number = config["laser_odometry"]["matching_method"]["correspond_number"];
                    if(in_cloud->size() < correspond_number || ref_cloud->size() < correspond_number)
                    {
                        std::cout << "Not Enough Correspondence:" << in_cloud->size() << "," << ref_cloud->size() << std::endl;
                        std::cout << "ICP Iterations Failed!!" << std::endl;
                        break;
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
                    saveMarkerToFile(normalMarker, OUTPUT_DIR + "ref_normal_markers/" + timestamp + "_" + std::to_string(i) + ".obj");

                    std::vector<Eigen::Vector3d> in_cloud_vec, ref_cloud_vec;
                    getXYZ(in_cloud, in_cloud_vec);
                    getXYZ(ref_cloud, ref_cloud_vec);
                    std::vector<Eigen::Vector3d> ref_normal;
                    getNormals(ref_cloud, ref_normal);

                    // 计算帧间位移．从当前的source -> target
                    Eigen::Matrix4d deltaTrans;


                    // Solve motion given matched points
                    std::string solve_method = config["laser_odometry"]["solve_method"]["method"];
                    bool flag = false;
                    
                    flag = solveMotionEstimationProblem(solve_method, in_cloud_vec, ref_cloud_vec, ref_normal, deltaTrans, timestamp);

                    if(flag == false)
                    {
                        std::cout << "Solve Failed!!!!" << std::endl;
                        std::cout << "ICP Iterations Failed!!!!"<< std::endl;
                        break;
                    }

                    // 更新位姿
                    rPose = deltaTrans * rPose;

                    // save matched points (ProjSourcePtToSurface return)
                    std::string filenameMP = OUTPUT_DIR + "matched_points/" + timestamp + "_" + std::to_string(i) + ".txt";
                    saveMatchedPointsToFile(in_cloud_vec, ref_cloud_vec, filenameMP);
                    // save pose
                    savePoseToFile(rPose, OUTPUT_DIR + "imls_iter_results.txt", timestamp);

                    // 计算平移距离
                    double deltaDist = std::sqrt(
                        std::pow(deltaTrans(0, 3), 2) + 
                        std::pow(deltaTrans(1, 3), 2) + 
                        std::pow(deltaTrans(2, 3), 2)
                    );

                    // 计算旋转角度
                    // 可以通过旋转矩阵的迹来计算旋转角的绝对值
                    double cos_theta = (deltaTrans.block<3,3>(0,0).trace() - 1.0) / 2.0;
                    cos_theta = std::min(1.0, std::max(cos_theta, -1.0)); // 确保cos_theta在[-1, 1]之间
                    double deltaAngle = std::acos(cos_theta);

                    double delta_dist_threshold = config["laser_odometry"]["solve_method"]["delta_dist_threshold"];
                    double delta_angle_threshold = config["laser_odometry"]["solve_method"]["delta_angle_threshold"]; // 0.01/57.295
                    // 如果迭代条件允许，则直接退出
                    if(deltaDist < delta_dist_threshold && deltaAngle < delta_angle_threshold)
                    {
                        break;
                    }
                }

                q_last_curr = rPose.block<3,3>(0,0);
                t_last_curr = rPose.block<3,1>(0,3); 

                Eigen::Matrix4d nowPose = prevLaserPose * rPose;
                q_global_curr = nowPose.block<3,3>(0,0);
                t_global_curr = nowPose.block<3,1>(0,3);
                prevLaserPose = nowPose;

                pubPath(t_global_curr, q_global_curr, odoPath, pubOdoPath, time);
                savePoseToFile(nowPose, pose_file, timestamp);

                t_step.tocAndLog("2. Matching and solving in flat points", timesFile); 


                /* -------------------- 3. matching using multiple frames --------------------*/


            }

            const size_t max_queue_size = config["laser_odometry"]["max_queue_size"];

            accumulateTargetCloud(filteredLaserCloud, max_queue_size, transform_normal);
            frameCount++;
            if (config["scan_registration"]["presample_method"]["method"]=="tensor_voting")
            {
                prevFilteredLaserCloudDP = filteredLaserCloudDP;
            }

            t_whole.tocAndLog("Total time", timesFile);
        }

        // Add a small delay to prevent this loop from hogging CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "laser_odometry");
    ros::NodeHandle nh;

    try {
        loadConfig();
        pose_file = OUTPUT_DIR + "imls_results.txt";
    } catch (const std::exception& e) {
        std::cerr << "加载配置文件失败: " << e.what() << std::endl;
        return -1;
    }

    pubOdoPath = nh.advertise<nav_msgs::Path>("imls_path_pub_", 1, true);
    odoPath.header.frame_id = "/camera_init";

    subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_filtered", 100, laserCloudHandler);
    subLaserFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserFlatHandler);
    subLaserLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserLessFlatHandler);
    subLaserCloudDP = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_filtered_dp", 100, laserCloudHandlerDP);
    subLaserFlatDP = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat_dp", 100, laserFlatHandlerDP);

    std::thread processThread(processData);
    ros::spin();
    processThread.join();

    return 0;
}
