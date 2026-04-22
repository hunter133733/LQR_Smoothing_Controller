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
    const rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
    nav_msgs::msg::Path path_;
    Eigen::MatrixXd K_;
    double max_linear_vel_;
    double max_angular_vel_;
};

}    // namespace lqr_nav2_controller