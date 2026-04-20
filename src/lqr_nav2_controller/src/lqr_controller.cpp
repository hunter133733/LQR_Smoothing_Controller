#include "lqr_nav2_controller/lqr_controller.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "angles/angles.h"
#include <tf2/utils.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace lqr_nav2_controller
{

void LQRController::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, 
    std::shared_ptr<tf2_ros::Buffer> /*tf*/,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> /*costmap_ros*/)
{
    auto node = parent.lock();

    // Basic velocity limits
    node->declare_parameter(name + ".max_linear_vel",  0.5);
    node->declare_parameter(name + ".max_angular_vel", 1.2);
    max_linear_vel_  = node->get_parameter(name + ".max_linear_vel").as_double();
    max_angular_vel_ = node->get_parameter(name + ".max_angular_vel").as_double();
 
    // LQR tuning parameters
    node->declare_parameter(name + ".dt",      0.1);
    node->declare_parameter(name + ".horizon", 40);
    node->declare_parameter(name + ".v_min",  -0.2);
    node->declare_parameter(name + ".v_max",   1.0);
    node->declare_parameter(name + ".w_min",  -1.2);
    node->declare_parameter(name + ".w_max",   1.2);
 
    dt_      = node->get_parameter(name + ".dt").as_double();
    horizon_ = node->get_parameter(name + ".horizon").as_int();
    v_min_   = node->get_parameter(name + ".v_min").as_double();
    v_max_   = node->get_parameter(name + ".v_max").as_double();
    w_min_   = node->get_parameter(name + ".w_min").as_double();
    w_max_   = node->get_parameter(name + ".w_max").as_double();
 
    // Cost coefficients (matches Python DEFAULT_COST_COEFS)
    node->declare_parameter(name + ".cost_x",     5.0);
    node->declare_parameter(name + ".cost_y",     5.0);
    node->declare_parameter(name + ".cost_theta", 0.3);
    node->declare_parameter(name + ".cost_v",     0.3);
    node->declare_parameter(name + ".cost_w",     0.3);
    node->declare_parameter(name + ".cost_dv",    0.05);
    node->declare_parameter(name + ".cost_dw",    0.10);
 
    double cx  = node->get_parameter(name + ".cost_x").as_double();
    double cy  = node->get_parameter(name + ".cost_y").as_double();
    double cth = node->get_parameter(name + ".cost_theta").as_double();
    double cv  = node->get_parameter(name + ".cost_v").as_double();
    double cw  = node->get_parameter(name + ".cost_w").as_double();
    double cdv = node->get_parameter(name + ".cost_dv").as_double();
    double cdw = node->get_parameter(name + ".cost_dw").as_double();
 
    // State cost matrices
    Qz_ = Eigen::Vector3d(cx, cy, cth).asDiagonal();
    Lz_ = Qz_;  // terminal = running (same as Python)
 
    // Control cost matrices
    Ru_  = Eigen::Vector2d(cv,  cw).asDiagonal();
    Rdu_ = Eigen::Vector2d(cdv, cdw).asDiagonal();
 
    // Augmented cost matrices (5x5): [delta_z (3); delta_u_prev (2)]
    Q_aug_.setZero();
    Q_aug_.block<3,3>(0,0) = Qz_;
    Q_aug_.block<2,2>(3,3) = Ru_;
 
    L_aug_.setZero();
    L_aug_.block<3,3>(0,0) = Lz_;
    L_aug_.block<2,2>(3,3) = Ru_;
 
    // Reset receding-horizon state
    delta_u_prev_.setZero();
 
    // Unused matrix kept for ABI compatibility
    K_ = Eigen::MatrixXd::Zero(2, 3);
 
    RCLCPP_INFO(node->get_logger(),
        "SmoothedLQRController configured: dt=%.2f horizon=%d", dt_, horizon_);
}

void LQRController::cleanup()
{}

void LQRController::activate()
{}

void LQRController::deactivate()
{}

void LQRController::setPlan(const nav_msgs::msg::Path & path)
{
    if (path_.poses.empty()) {
        delta_u_prev_.setZero(); //reset if new path
    }
    path_ = path;
}

// ===========================================================================
// Dubins car linearization  (forward-Euler discrete time)
//   x_dot     = v*cos(theta)
//   y_dot     = v*sin(theta)
//   theta_dot = w
// ===========================================================================
void LQRController::linearize(
    const Eigen::Vector3d & z,
    const Eigen::Vector2d & u,
    double dt,
    Eigen::Matrix3d & A,
    Eigen::Matrix<double, 3, 2> & B) const
{
    double theta = z(2);
    double v     = u(0);
 
    // Continuous-time Jacobians
    Eigen::Matrix3d Ac = Eigen::Matrix3d::Zero();
    Ac(0, 2) = -v * std::sin(theta);
    Ac(1, 2) =  v * std::cos(theta);
 
    Eigen::Matrix<double, 3, 2> Bc = Eigen::Matrix<double, 3, 2>::Zero();
    Bc(0, 0) = std::cos(theta);
    Bc(1, 0) = std::sin(theta);
    Bc(2, 1) = 1.0;
 
    // Forward-Euler discretization
    A = Eigen::Matrix3d::Identity() + dt * Ac;
    B = dt * Bc;
}
 
// ===========================================================================
// Finite-horizon time-varying LQR gains (augmented Ext3 system)
//
// Augmented state:  xi_t  = [delta_z_t (3); delta_u_{t-1} (2)]   (5-dim)
// New control:      nu_t  = delta_u_t - delta_u_{t-1}             (2-dim)
//
// xi_{t+1} = A'_t xi_t + B'_t nu_t
//   A'_t = [ A_t  B_t ]     B'_t = [ B_t ]
//          [  0    I  ]             [  I  ]
//
// Cost:  sum Q_aug + terminal L_aug,  change cost Rdu on nu_t
// ===========================================================================
std::vector<Eigen::Matrix<double, 2, 5>> LQRController::computeGains(
    const std::vector<Eigen::Matrix3d> & As,
    const std::vector<Eigen::Matrix<double, 3, 2>> & Bs) const
{
    int n = static_cast<int>(As.size());
 
    // Build augmented system matrices
    using A5 = Eigen::Matrix<double, 5, 5>;
    using B5 = Eigen::Matrix<double, 5, 2>;
    using K5 = Eigen::Matrix<double, 2, 5>;
 
    std::vector<A5> A_aug(n);
    std::vector<B5> B_aug(n);
 
    for (int i = 0; i < n; ++i) {
        A_aug[i].setZero();
        A_aug[i].block<3,3>(0,0) = As[i];
        A_aug[i].block<3,2>(0,3) = Bs[i];
        A_aug[i].block<2,2>(3,3) = Eigen::Matrix2d::Identity();
 
        B_aug[i].setZero();
        B_aug[i].block<3,2>(0,0) = Bs[i];
        B_aug[i].block<2,2>(3,0) = Eigen::Matrix2d::Identity();
    }
 
    // Backward Riccati
    std::vector<K5> Ks(n);
    A5 P = L_aug_;  // terminal cost
 
    for (int i = n - 1; i >= 0; --i) {
        const A5 & At = A_aug[i];
        const B5 & Bt = B_aug[i];
 
        // S = Rdu + B'P B  (2x2)
        Eigen::Matrix2d S = Rdu_ + Bt.transpose() * P * Bt;
 
        // K = S^{-1} B'P A  (2x5)
        Ks[i] = S.ldlt().solve(Bt.transpose() * P * At);
 
        // P = Q_aug + A'PA - A'PB K
        A5 P_new = Q_aug_ + At.transpose() * P * At - At.transpose() * P * Bt * Ks[i];
 
        // Enforce symmetry for numerical stability
        P = 0.5 * (P_new + P_new.transpose());
    }
 
    return Ks;
}
 
// ===========================================================================
// One receding-horizon solve step
// ===========================================================================
Eigen::Vector2d LQRController::solve(
    const Eigen::Vector3d & z0,
    const std::vector<Eigen::Vector3d> & z_ref,
    const std::vector<Eigen::Vector2d> & u_ref)
{
    int n_track = static_cast<int>(std::min({
        static_cast<size_t>(horizon_),
        z_ref.size(),
        u_ref.size()}));
 
    if (n_track == 0) return Eigen::Vector2d::Zero();
 
    // Linearize along reference trajectory
    std::vector<Eigen::Matrix3d>          As(n_track);
    std::vector<Eigen::Matrix<double,3,2>> Bs(n_track);
    for (int i = 0; i < n_track; ++i) {
        linearize(z_ref[i], u_ref[i], dt_, As[i], Bs[i]);
    }
 
    // Compute gains
    auto Ks = computeGains(As, Bs);
 
    // Simulate forward to get first control
    Eigen::Vector3d z_t = z0;
    Eigen::Vector2d delta_u_prev = delta_u_prev_;
    Eigen::Vector2d u_first = Eigen::Vector2d::Zero();
 
    for (int i = 0; i < n_track; ++i) {
        // State error
        Eigen::Vector3d delta_z = z_t - z_ref[i];
        delta_z(2) = wrapAngle(delta_z(2));
 
        // Augmented state: xi = [delta_z; delta_u_prev]
        Eigen::Matrix<double, 5, 1> xi;
        xi.head<3>() = delta_z;
        xi.tail<2>() = delta_u_prev;
 
        // nu = increment in control deviation
        Eigen::Vector2d nu = -Ks[i] * xi;
 
        // Recover control deviation and actual control
        Eigen::Vector2d delta_u = delta_u_prev + nu;
        Eigen::Vector2d u_t = u_ref[i] + delta_u;
 
        // Clamp to actuator limits
        u_t(0) = std::clamp(u_t(0), v_min_, v_max_);
        u_t(1) = std::clamp(u_t(1), w_min_, w_max_);
 
        // Keep deviation consistent with clipped action
        delta_u = u_t - u_ref[i];
        delta_u_prev = delta_u;
 
        if (i == 0) u_first = u_t;
 
        // Propagate Dubins dynamics (forward Euler)
        double theta = z_t(2);
        z_t(0) += dt_ * u_t(0) * std::cos(theta);
        z_t(1) += dt_ * u_t(0) * std::sin(theta);
        z_t(2)  = wrapAngle(z_t(2) + dt_ * u_t(1));
    }
 
    // Update smoothing state for next call
    delta_u_prev_ = delta_u_prev;
 
    return u_first;
}
 
// ===========================================================================
// Build reference trajectory from the Nav2 path
// Estimates reference velocities from path geometry
// ===========================================================================
void LQRController::buildReference(
    const nav_msgs::msg::Path & path,
    const Eigen::Vector3d & z0,
    std::vector<Eigen::Vector3d> & z_ref,
    std::vector<Eigen::Vector2d> & u_ref) const
{
    if (path.poses.empty()) return;
 
    // Find closest point on path
    size_t start_idx = 0;
    double best = std::numeric_limits<double>::max();
    for (size_t i = 0; i < path.poses.size(); ++i) {
        double dx = z0(0) - path.poses[i].pose.position.x;
        double dy = z0(1) - path.poses[i].pose.position.y;
        double d  = dx*dx + dy*dy;
        if (d < best) { best = d; start_idx = i; }
    }
 
    // Extract up to horizon_ + 1 waypoints starting from closest
    size_t n_pts = std::min(
        static_cast<size_t>(horizon_ + 1),
        path.poses.size() - start_idx);
 
    z_ref.clear();
    u_ref.clear();
 
    for (size_t k = 0; k < n_pts; ++k) {
        size_t idx = start_idx + k;
        const auto & p = path.poses[idx].pose;
 
        Eigen::Vector3d z;
        z(0) = p.position.x;
        z(1) = p.position.y;
 
        // Heading from path orientation or from segment direction
        if (idx + 1 < path.poses.size()) {
            const auto & pn = path.poses[idx + 1].pose;
            z(2) = std::atan2(
                pn.position.y - p.position.y,
                pn.position.x - p.position.x);
        } else {
            z(2) = tf2::getYaw(p.orientation);
        }
 
        z_ref.push_back(z);
 
        // Reference control: estimate from consecutive waypoints
        if (idx + 1 < path.poses.size()) {
            const auto & pn = path.poses[idx + 1].pose;
            double seg_len = std::hypot(
                pn.position.x - p.position.x,
                pn.position.y - p.position.y);
            double v_ref = std::clamp(seg_len / dt_, 0.05, v_max_);
 
            double theta_next = (idx + 2 < path.poses.size()) ?
                std::atan2(
                    path.poses[idx+2].pose.position.y - pn.position.y,
                    path.poses[idx+2].pose.position.x - pn.position.x) :
                z(2);
            double w_ref = std::clamp(
                wrapAngle(theta_next - z(2)) / dt_,
                w_min_, w_max_);
 
            u_ref.push_back(Eigen::Vector2d(v_ref, w_ref));
        } else {
            u_ref.push_back(Eigen::Vector2d::Zero());
        }
    }
}

// ===========================================================================
// computeVelocityCommands
// ===========================================================================
geometry_msgs::msg::TwistStamped LQRController::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & /*velocity*/,
    nav2_core::GoalChecker * /*goal_checker*/)
{
    geometry_msgs::msg::TwistStamped cmd_vel;
    cmd_vel.header = pose.header;

    if (path_.poses.empty()) return cmd_vel;

    // Current robot state
    Eigen::Vector3d z0;
    z0(0) = pose.pose.position.x;
    z0(1) = pose.pose.position.y;
    z0(2) = tf2::getYaw(pose.pose.orientation);
 
    // Build reference window
    std::vector<Eigen::Vector3d> z_ref;
    std::vector<Eigen::Vector2d> u_ref;
    buildReference(path_, z0, z_ref, u_ref);
 
    if (z_ref.empty()) return cmd_vel;
 
    // Run smoothed LQR
    Eigen::Vector2d u = solve(z0, z_ref, u_ref);
 
    // Clamp to nav2 speed limits
    u(0) = std::clamp(u(0), -max_linear_vel_,  max_linear_vel_);
    u(1) = std::clamp(u(1), -max_angular_vel_, max_angular_vel_);
 
    cmd_vel.twist.linear.x  = u(0);
    cmd_vel.twist.angular.z = u(1);
 
    return cmd_vel;
}

void LQRController::setSpeedLimit(
    const double & speed_limit, 
    const bool & percentage)
{
    if (percentage) {
        max_linear_vel_ = max_linear_vel_ * speed_limit / 100.0;
    } else {
        max_linear_vel_ = speed_limit;
    }
}

}  // namespace lqr_nav2_controller

PLUGINLIB_EXPORT_CLASS(
  lqr_nav2_controller::LQRController,
  nav2_core::Controller
)