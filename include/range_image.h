#ifndef RANGE_IMAGE_H
#define RANGE_IMAGE_H

#include <Eigen/Dense>
#include <vector>

// 定义点结构
struct PointT {
    float x, y, z;
};

class RangeImage {
public:
    // 构造函数
    RangeImage(int width = 4000, int height = 64, float f_up = 2.0f, float f_down = -24.33f);

    bool computeNormalFALS(
        const Eigen::MatrixXf& range_image, 
        int window_size,
        std::vector<int>& indices_list, 
        std::vector<Eigen::Vector3f>& normal_list);

    bool computeNormalSRI(
        const Eigen::MatrixXf& range_image, 
        int window_size,
        std::vector<int>& indices_list, 
        std::vector<Eigen::Vector3f>& normal_list);

    bool computeCurvature(
        const Eigen::MatrixXf& range_image,
        int window_size,
        Eigen::MatrixXf& curvature_map);

private:
    // 类参数
    int width_;               // 图像宽度
    int height_;              // 图像高度
    float f_up_;              // 垂直视场角上界
    float f_down_;            // 垂直视场角下界
    float fov_up_rad_;        // 上视场角（弧度）
    float fov_down_rad_;      // 下视场角（弧度）
    float fov_total_;         // 总视场角

    // 静态成员变量
    static Eigen::MatrixXf azimuth_;      // 水平角矩阵
    static Eigen::MatrixXf vertical_;    // 垂直角矩阵
    static std::vector<std::vector<Eigen::Matrix3f>> M_inv_map_;
    static std::vector<std::vector<Eigen::Matrix3f>> rhat_map_;
    static bool angle_matrices_initialized_, M_inv_initialized_, rhat_initialized_; // 标记是否已初始化
    
    // 初始化角度矩阵
    void initializeAngleMatrices();

    // 初始化 M_inv 矩阵
    void initializeMInv(int window_size);

    // 初始化 Rhat 矩阵
    void initializeRhat();
};

#endif // RANGE_IMAGE_H
