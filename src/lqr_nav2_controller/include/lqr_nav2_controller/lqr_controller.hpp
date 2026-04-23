#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav2_core/controller.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>

namespace lqr_nav2_controller
{

class LQRController : public nav2_core::Controller
{
public:
    LQRController() = default;
    ~LQRController() override = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name,
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

    void cleanup()    override;
    void activate()   override;
    void deactivate() override;

    void setPlan(const nav_msgs::msg::Path & path) override;

    geometry_msgs::msg::TwistStamped computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist       & velocity,
        nav2_core::GoalChecker * goal_checker) override;

    void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
    void linearize(
        const Eigen::Vector3d     & z,
        const Eigen::Vector2d     & u,
        double                      dt,
        Eigen::Matrix3d           & A,
        Eigen::Matrix<double,3,2> & B) const;

    void buildAugmentedDynamics(
        const std::vector<Eigen::Matrix3d>           & As,
        const std::vector<Eigen::Matrix<double,3,2>> & Bs,
        std::vector<Eigen::Matrix<double,5,5>>       & A_aug,
        std::vector<Eigen::Matrix<double,5,2>>       & B_aug) const;

    std::vector<Eigen::Matrix<double,2,5>> computeGains(
        const std::vector<Eigen::Matrix<double,5,5>> & A_aug,
        const std::vector<Eigen::Matrix<double,5,2>> & B_aug) const;

    Eigen::Vector2d solve(
        const Eigen::Vector3d              & z0,
        const std::vector<Eigen::Vector3d> & z_ref,
        const std::vector<Eigen::Vector2d> & u_ref,
        const Eigen::Vector2d              & u_prev_ref_0);

    void buildReference(
        const nav_msgs::msg::Path    & path,
        const Eigen::Vector3d        & z0,
        std::vector<Eigen::Vector3d> & z_ref,
        std::vector<Eigen::Vector2d> & u_ref,
        Eigen::Vector2d              & u_prev_ref_0) const;

    double wrapAngle(double a) const
    {
        return std::atan2(std::sin(a), std::cos(a));
    }

    // path
    nav_msgs::msg::Path path_;

    // controller state
    Eigen::Vector2d u_prev_{Eigen::Vector2d::Zero()};

    // parameters
    double dt_     {0.1};
    int    horizon_ {25};
    double v_min_ {-0.2};
    double v_max_  {1.0};
    double w_min_ {-1.2};
    double w_max_  {1.2};

    // cost matrices
    Eigen::Matrix3d           Qz_;    // 3x3 state stage cost
    Eigen::Matrix3d           Lz_;    // 3x3 state terminal cost
    Eigen::Matrix2d           Ru_;    // 2x2 absolute-velocity cost
    Eigen::Matrix2d           Rdu_;   // 2x2 control-rate cost
    Eigen::Matrix<double,5,5> Q_aug_; // 5x5 augmented stage cost
    Eigen::Matrix<double,5,5> L_aug_; // 5x5 augmented terminal cost
};

} // namespace lqr_nav2_controller