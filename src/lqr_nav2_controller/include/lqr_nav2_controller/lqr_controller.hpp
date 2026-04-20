#pragma once

#include "nav2_core/controller.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "tf2_ros/buffer.h"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include <Eigen/Dense>

namespace lqr_nav2_controller
{

class LQRController : public nav2_core::Controller
{
public:
    LQRController() = default;

    ~LQRController() {}

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr &,
        std::string /*name*/, 
        std::shared_ptr<tf2_ros::Buffer>,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS>) override;

    void cleanup() override;
    void activate() override;
    void deactivate() override;
    void setPlan(const nav_msgs::msg::Path & /*path*/) override;

    geometry_msgs::msg::TwistStamped computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & /*velocity*/,
        nav2_core::GoalChecker * /*goal_checker*/) override;

    void setSpeedLimit(const double & /*speed_limit*/, const bool & /*percentage*/) override;

private:
    // ---------------------------------------------------------------
    // Dubins car linearization
    // Returns discrete-time A (3x3) and B (3x2) via forward Euler
    // ---------------------------------------------------------------
    void linearize(
        const Eigen::Vector3d & z,
        const Eigen::Vector2d & u,
        double dt,
        Eigen::Matrix3d & A,
        Eigen::Matrix<double, 3, 2> & B) const;
 
    // ---------------------------------------------------------------
    // Compute finite-horizon time-varying LQR gains (augmented Ext3)
    // As: (n, 3x3), Bs: (n, 3x2)
    // Returns Ks: (n, 2x5)
    // ---------------------------------------------------------------
    std::vector<Eigen::Matrix<double, 2, 5>> computeGains(
        const std::vector<Eigen::Matrix3d> & As,
        const std::vector<Eigen::Matrix<double, 3, 2>> & Bs) const;
 
    // ---------------------------------------------------------------
    // Run one receding-horizon solve from z0 tracking z_ref / u_ref
    // Returns first control action
    // ---------------------------------------------------------------
    Eigen::Vector2d solve(
        const Eigen::Vector3d & z0,
        const std::vector<Eigen::Vector3d> & z_ref,
        const std::vector<Eigen::Vector2d> & u_ref);
 
    // ---------------------------------------------------------------
    // Build reference trajectory from nav_msgs::Path
    // ---------------------------------------------------------------
    void buildReference(
        const nav_msgs::msg::Path & path,
        const Eigen::Vector3d & z0,
        std::vector<Eigen::Vector3d> & z_ref,
        std::vector<Eigen::Vector2d> & u_ref) const;
 
    static double wrapAngle(double a)
    {
        return std::atan2(std::sin(a), std::cos(a));
    }
 
    // ---- nav2 interface state ----
    nav_msgs::msg::Path path_;
    double max_linear_vel_{0.5};
    double max_angular_vel_{1.2};
 
    // ---- LQR parameters ----
    double dt_{0.1};
    int    horizon_{25};
    double v_min_{-0.2}, v_max_{1.0};
    double w_min_{-1.2}, w_max_{1.2};
 
    // ---- Cost matrices ----
    // State cost Q (3x3) and terminal L (3x3)
    Eigen::Matrix3d Qz_, Lz_;
    // Control deviation cost Ru (2x2), change-in-deviation cost Rdu (2x2)
    Eigen::Matrix2d Ru_, Rdu_;
    // Augmented (5x5) costs
    Eigen::Matrix<double, 5, 5> Q_aug_, L_aug_;
 
    // ---- Receding horizon state ----
    Eigen::Vector2d delta_u_prev_{Eigen::Vector2d::Zero()};
 
    // ---- Unused K_ kept for ABI compatibility with base header ----
    Eigen::MatrixXd K_;
};

}    // namespace lqr_nav2_controller