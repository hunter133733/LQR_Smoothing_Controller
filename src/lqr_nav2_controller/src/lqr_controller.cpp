#include "lqr_nav2_controller/lqr_controller.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "angles/angles.h"
#include <algorithm>

namespace lqr_nav2_controller
{

void LQRController::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, 
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
    auto node = parent.lock();

    node->declare_parameter("max_linear_vel", 0.5);
    node->declare_parameter("max_angular_vel", 1.0);

    node->get_parameter("max_linear_vel", max_linear_vel_);
    node->get_parameter("max_angular_vel", max_angular_vel_);
}

void LQRController::cleanup()
{}

void LQRController::activate()
{}

void LQRController::deactivate()
{}

void LQRController::setPlan(const nav_msgs::msg::Path & path)
{
    path_ = path;
}

geometry_msgs::msg::TwistStamped LQRController::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & /*velocity*/,
    nav2_core::GoalChecker * /*goal_checker*/)
{
    geometry_msgs::msg::TwistStamped cmd_vel;
    cmd_vel.header = pose.header;

    // -------------------------------
    // 1. Current robot state
    // -------------------------------
    double x = pose.pose.position.x;
    double y = pose.pose.position.y;
    double theta = tf2::getYaw(pose.pose.orientation);

    // -------------------------------
    // 2. Find closest point on path
    // -------------------------------
    int closest_index = 0;
    double best_dist = std::numeric_limits<double>::max();

    for (size_t i = 0; i < path_.poses.size(); i++) {
        double dx = x - path_.poses[i].pose.position.x;
        double dy = y - path_.poses[i].pose.position.y;
        double dist = dx * dx + dy * dy;

        if (dist < best_dist) {
        best_dist = dist;
        closest_index = i;
        }
    }

    // -------------------------------
    // 3. Lookahead point
    // -------------------------------
    double lookahead_dist = 0.5; // tune this
    double accumulated_dist = 0.0;
    int target_index = closest_index;

    while (target_index < (int)path_.poses.size() - 1 &&
            accumulated_dist < lookahead_dist) {

        auto & p1 = path_.poses[target_index];
        auto & p2 = path_.poses[target_index + 1];

        accumulated_dist += hypot(
        p2.pose.position.x - p1.pose.position.x,
        p2.pose.position.y - p1.pose.position.y);

        target_index++;
    }

    auto & target_pose = path_.poses[target_index];

    double x_ref = target_pose.pose.position.x;
    double y_ref = target_pose.pose.position.y;

    // -------------------------------
    // 4. Reference heading (theta_ref)
    // -------------------------------
    double theta_ref;

    if (target_index < (int)path_.poses.size() - 1) {
        auto & next_pose = path_.poses[target_index + 1];

        theta_ref = atan2(
        next_pose.pose.position.y - y_ref,
        next_pose.pose.position.x - x_ref);
    } else {
        theta_ref = tf2::getYaw(target_pose.pose.orientation);
    }

    // -------------------------------
    // 5. Compute error (state difference)
    // -------------------------------
    Eigen::Vector3d error;

    double dx = x - x_ref;
    double dy = y - y_ref;

    // Rotate error into robot frame (IMPORTANT)
    error(0) =  cos(theta) * dx + sin(theta) * dy;   // forward error
    error(1) = -sin(theta) * dx + cos(theta) * dy;   // lateral error
    error(2) = angles::shortest_angular_distance(theta, theta_ref);

    // -------------------------------
    // 6. LQR control
    // -------------------------------
    Eigen::Vector2d u = -K_ * error;

    double v = u(0);
    double omega = u(1);

    // -------------------------------
    // 7. Clamp velocities
    // -------------------------------
    v = std::clamp(v, -max_linear_vel_, max_linear_vel_);
    omega = std::clamp(omega, -max_angular_vel_, max_angular_vel_);

    // -------------------------------
    // 8. Fill command
    // -------------------------------
    cmd_vel.twist.linear.x = v;
    cmd_vel.twist.angular.z = omega;

    return cmd_vel;
}

void LQRController::setSpeedLimit(const double & /*speed_limit*/, const bool & /*percentage*/)
{}

}  // namespace lqr_nav2_controller

PLUGINLIB_EXPORT_CLASS(
  lqr_nav2_controller::LQRController,
  nav2_core::Controller
)