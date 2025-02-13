#include "range_image.h"
#include <cmath>
#include <limits>
#include <algorithm>


Eigen::MatrixXf RangeImage::azimuth_;
Eigen::MatrixXf RangeImage::vertical_;
std::vector<std::vector<Eigen::Matrix3f>> RangeImage::M_inv_map_;
std::vector<std::vector<Eigen::Matrix3f>> RangeImage::rhat_map_;
bool RangeImage::angle_matrices_initialized_ = false;
bool RangeImage::M_inv_initialized_ = false;
bool RangeImage::rhat_initialized_ = false;


RangeImage::RangeImage(int width, int height, float f_up, float f_down)
    : width_(width), height_(height), f_up_(f_up), f_down_(f_down) {
    if (!angle_matrices_initialized_) {
        initializeAngleMatrices();
        angle_matrices_initialized_ = true;
    }
}

void RangeImage::initializeAngleMatrices() {
    fov_up_rad_ = f_up_ * M_PI / 180.0f;
    fov_down_rad_ = f_down_ * M_PI / 180.0f;
    fov_total_ = fov_up_rad_ - fov_down_rad_;

    azimuth_.resize(height_, width_);
    vertical_.resize(height_, width_);

    for (int row = 0; row < height_; ++row) {
        for (int col = 0; col < width_; ++col) {
            azimuth_(row, col) = 2 * M_PI * (1 - static_cast<float>(col) / width_) - M_PI;
            vertical_(row, col) = fov_down_rad_ + fov_total_ * (1 - static_cast<float>(row) / height_);
        }
    }
}

void RangeImage::initializeMInv(int window_size) {
    // 创建一个三维矩阵用于存储每个像素位置的 M^-1 矩阵
    M_inv_map_.resize(height_, std::vector<Eigen::Matrix3f>(width_));

    // 遍历整个 range image
    for (int row = 0; row < height_; ++row) {
        for (int col = 0; col < width_; ++col) {
            Eigen::Matrix3f M = Eigen::Matrix3f::Zero();

            // 遍历窗口内的点
            for (int row_offset = -window_size; row_offset <= window_size; ++row_offset) {
                for (int col_offset = -window_size; col_offset <= window_size; ++col_offset) {
                    int neighbor_row = row + row_offset;
                    int neighbor_col = col + col_offset;

                    // 检查是否越界
                    if (neighbor_row < 0 || neighbor_row >= height_ || neighbor_col < 0 || neighbor_col >= width_) {
                        continue;
                    }

                    // 获取对应的 azimuth 和 vertical 角度
                    float theta = azimuth_(neighbor_row, neighbor_col);
                    float phi = vertical_(neighbor_row, neighbor_col);

                    // 计算 v 向量
                    Eigen::Vector3f v;
                    v << std::sin(theta) * std::cos(phi),
                         std::sin(phi),
                         std::cos(theta) * std::cos(phi);

                    // 累加到 M
                    M += v * v.transpose();
                }
            }

            // 计算并存储 M^-1
            if (M.determinant() > 1e-6) {  // 检查 M 是否可逆
                M_inv_map_[row][col] = M.inverse();
            } else {
                M_inv_map_[row][col] = Eigen::Matrix3f::Zero();  // 若不可逆，存储零矩阵
            }
        }
    }
    M_inv_initialized_ = true;
}

void RangeImage::initializeRhat() {
    rhat_map_.resize(height_, std::vector<Eigen::Matrix3f>(width_));
    for (int row = 0; row < height_; ++row) {
        for (int col = 0; col < width_; ++col) {
            float theta = azimuth_(row, col);
            float phi = vertical_(row, col);

            // 计算 R_θ,φ
            Eigen::Matrix3f R_theta_phi;
            R_theta_phi << std::cos(theta), -std::sin(theta), 0,
                           std::sin(theta),  std::cos(theta), 0,
                           0,                0,               1;

            Eigen::Matrix3f R_phi;
            R_phi << std::cos(phi), 0, -std::sin(phi),
                     0,            1,  0,
                     std::sin(phi), 0,  std::cos(phi);

            Eigen::Matrix3f R = R_theta_phi * R_phi;

            // 计算 Rhat
            Eigen::Matrix3f rhat;
            rhat.col(0) = Eigen::Vector3f(0, 0, 1);  // zhat
            rhat.col(1) = Eigen::Vector3f(1, 0, 0);  // xhat
            rhat.col(2) = Eigen::Vector3f(0, 1, 0);  // yhat
            rhat_map_[row][col] = rhat * R;
        }
    }
    rhat_initialized_ = true;
}

bool RangeImage::computeNormalFALS(
    const Eigen::MatrixXf& range_image,
    int window_size,
    std::vector<int>& indices_list,
    std::vector<Eigen::Vector3f>& normal_list) {
    indices_list.clear();
    normal_list.clear();

    if (!M_inv_initialized_) {
        initializeMInv(window_size);
    }

    int global_index = -1;

    // 遍历每个点计算法向量
    for (int row = 0; row < height_; ++row) {
        for (int col = 0; col < width_; ++col) {
            if (range_image(row, col) == std::numeric_limits<float>::infinity()) continue;
            global_index++;

            Eigen::Vector3f b = Eigen::Vector3f::Zero();

            // 遍历窗口内的点计算 b
            for (int r_offset = -window_size; r_offset <= window_size; ++r_offset) {
                for (int c_offset = -window_size; c_offset <= window_size; ++c_offset) {
                    int neighbor_row = row + r_offset;
                    int neighbor_col = col + c_offset;

                    // 检查是否越界
                    if (neighbor_row < 0 || neighbor_row >= height_ || neighbor_col < 0 || neighbor_col >= width_) {
                        continue;
                    }

                    float range = range_image(neighbor_row, neighbor_col);
                    if (range == std::numeric_limits<float>::infinity()) continue;

                    float theta = azimuth_(neighbor_row, neighbor_col);
                    float phi = vertical_(neighbor_row, neighbor_col);

                    Eigen::Vector3f v;
                    v << std::sin(theta) * std::cos(phi),
                         std::sin(phi),
                         std::cos(theta) * std::cos(phi);

                    b += v / range;
                }
            }

            // 使用预计算的 M^-1 矩阵计算法向量
            Eigen::Matrix3f M_inv = M_inv_map_[row][col];
            if (M_inv.isZero()) continue;  // 如果 M^-1 矩阵无效，跳过此点

            Eigen::Vector3f normal = M_inv * b;
            normal.normalize();

            // 存储法向量和全局索引
            normal_list.push_back(normal);

            indices_list.push_back(global_index);
        }
    }

    return true;
}

bool RangeImage::computeNormalSRI(
    const Eigen::MatrixXf& range_image, 
    int window_size,
    std::vector<int>& indices_list, 
    std::vector<Eigen::Vector3f>& normal_list) {
    normal_list.clear();
    indices_list.clear();

    if (!rhat_initialized_) {
        initializeRhat();
    }

    int global_index = -1;

    // 动态生成 Prewitt 算子
    Eigen::MatrixXf Mx = Eigen::MatrixXf::Zero(2 * window_size + 1, 2 * window_size + 1);
    Eigen::MatrixXf My = Eigen::MatrixXf::Zero(2 * window_size + 1, 2 * window_size + 1);

    // debug and check!
    for (int i = -window_size; i <= window_size; ++i) {
        for (int j = -window_size; j <= window_size; ++j) {
            if (i < 0) {
                My(i + window_size, j + window_size) = +1;
            } else if (i > 0) {
                My(i + window_size, j + window_size) = -1;
            }

            if (j < 0) {
                Mx(i + window_size, j + window_size) = +1;
            } else if (j > 0) {
                Mx(i + window_size, j + window_size) = -1;
            }
        }
    }

    // 遍历每个点计算法向量
    for (int row = window_size; row < height_ - window_size; ++row) {
        for (int col = window_size; col < width_ - window_size; ++col) {
            float r = range_image(row, col);
            if (r == std::numeric_limits<float>::infinity()) continue;
            global_index++;

            // 计算梯度
            float dr_dtheta = 0.0f;
            float dr_dphi = 0.0f;

            for (int i = -window_size; i <= window_size; ++i) {
                for (int j = -window_size; j <= window_size; ++j) {
                    int neighbor_row = row + i;
                    int neighbor_col = col + j;

                    float neighbor_r = range_image(neighbor_row, neighbor_col);
                    if (neighbor_r == std::numeric_limits<float>::infinity()) continue;

                    dr_dtheta += neighbor_r * Mx(i + window_size, j + window_size);
                    dr_dphi += neighbor_r * My(i + window_size, j + window_size);
                }
            }

            float phi = vertical_(row, col);

            // 梯度向量
            Eigen::Vector3f grad;
            grad << 1.0f,
                    dr_dtheta / (r * std::cos(phi)),
                    dr_dphi / r;

            // 法向量计算：n = Rhat * grad
            Eigen::Matrix3f rhat = rhat_map_[row][col];
            Eigen::Vector3f normal = rhat * grad;
            normal.normalize();

            // 存储结果
            normal_list.push_back(normal);
            indices_list.push_back(global_index);
        }
    }

    return true;
}

bool RangeImage::computeCurvature(
    const Eigen::MatrixXf& range_image,
    int window_size,
    Eigen::MatrixXf& curvature_map) {

    curvature_map = Eigen::MatrixXf::Zero(height_, width_);

    // 遍历每个点计算曲率
    for (int row = window_size; row < height_ - window_size; ++row) {
        for (int col = window_size; col < width_ - window_size; ++col) {
            float r_center = range_image(row, col);
            if (r_center == std::numeric_limits<float>::infinity()) continue;

            float diffX = 0.0f, diffY = 0.0f, diffZ = 0.0f;

            // 累加窗口内的点
            for (int i = -window_size; i <= window_size; ++i) {
                // for (int j = -window_size; j <= window_size; ++j) {
                    int neighbor_row = row + i;
                    // int neighbor_col = col + j;
                    int neighbor_col = col;

                    // 检查邻域点是否有效
                    float r_neighbor = range_image(neighbor_row, neighbor_col);
                    if (r_neighbor == std::numeric_limits<float>::infinity()) continue;

                    // 计算邻域点的方向向量
                    float theta_neighbor = azimuth_(neighbor_row, neighbor_col);
                    float phi_neighbor = vertical_(neighbor_row, neighbor_col);

                    Eigen::Vector3f neighbor_point;
                    neighbor_point << r_neighbor * std::cos(phi_neighbor) * std::cos(theta_neighbor),
                                      r_neighbor * std::cos(phi_neighbor) * std::sin(theta_neighbor),
                                      r_neighbor * std::sin(phi_neighbor);

                    // 计算中心点的方向向量
                    float theta_center = azimuth_(row, col);
                    float phi_center = vertical_(row, col);

                    Eigen::Vector3f center_point;
                    center_point << r_center * std::cos(phi_center) * std::cos(theta_center),
                                    r_center * std::cos(phi_center) * std::sin(theta_center),
                                    r_center * std::sin(phi_center);

                    // 计算差值
                    Eigen::Vector3f diff = neighbor_point - center_point;

                    diffX += diff.x();
                    diffY += diff.y();
                    diffZ += diff.z();
                // }
            }

            // 计算曲率：平方和
            curvature_map(row, col) = diffX * diffX + diffY * diffY + diffZ * diffZ;
        }
    }

    return true;
}

