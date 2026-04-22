#include "lqr_nav2_controller/lqr_controller.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lqr_nav2_controller
{

// ================= CONFIGURE =================
void LQRController::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer>,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS>)
{
    auto node = parent.lock();

    //read parameters
    node->declare_parameter(name + ".cost_x", 5.0);
    node->declare_parameter(name + ".cost_y", 5.0);
    node->declare_parameter(name + ".cost_theta", 2.0);
    node->declare_parameter(name + ".cost_v", 0.5);
    node->declare_parameter(name + ".cost_w", 0.2);
    node->declare_parameter(name + ".cost_dv", 0.05);
    node->declare_parameter(name + ".cost_dw", 0.10);

    double cx  = node->get_parameter(name + ".cost_x").as_double();
    double cy  = node->get_parameter(name + ".cost_y").as_double();
    double cth = node->get_parameter(name + ".cost_theta").as_double();
    double cv  = node->get_parameter(name + ".cost_v").as_double();
    double cw  = node->get_parameter(name + ".cost_w").as_double();
    double cdv = node->get_parameter(name + ".cost_dv").as_double();
    double cdw = node->get_parameter(name + ".cost_dw").as_double();

    // state cost
    Qz_ = Eigen::Vector3d(cx, cy, cth).asDiagonal();
    Lz_ = Qz_;

    // control cost
    Ru_  = Eigen::Vector2d(cv, cw).asDiagonal();
    Rdu_ = Eigen::Vector2d(cdv, cdw).asDiagonal();

    // ===== AUGMENTED COST (5D state) =====
    Q_aug_.setZero();
    Q_aug_.block<3,3>(0,0) = Qz_;
    Q_aug_.block<2,2>(3,3) = Ru_;

    L_aug_ = Q_aug_;   // terminal cost same

    delta_u_prev_.setZero();
}

void LQRController::cleanup() {}
void LQRController::activate() {}
void LQRController::deactivate() {}

void LQRController::setPlan(const nav_msgs::msg::Path & path)
{
    path_ = path;
}

// ================= LINEARIZATION =================
void LQRController::linearize(
    const Eigen::Vector3d & z,
    const Eigen::Vector2d & u,
    double dt,
    Eigen::Matrix3d & A,
    Eigen::Matrix<double,3,2> & B) const
    {
    double th = z(2);
    double v = u(0);

    Eigen::Matrix3d Ac = Eigen::Matrix3d::Zero();
    Ac(0,2) = -v * std::sin(th);
    Ac(1,2) =  v * std::cos(th);

    Eigen::Matrix<double,3,2> Bc = Eigen::Matrix<double,3,2>::Zero();
    Bc(0,0) = std::cos(th);
    Bc(1,0) = std::sin(th);
    Bc(2,1) = 1.0;

    A = Eigen::Matrix3d::Identity() + dt * Ac;
    B = dt * Bc;
}

// ================= AUGMENTED LQR =================
std::vector<Eigen::Matrix<double,2,5>> LQRController::computeGains(
    const std::vector<Eigen::Matrix3d> & As,
    const std::vector<Eigen::Matrix<double,3,2>> & Bs) const
{
    int N = As.size();

    using A5 = Eigen::Matrix<double,5,5>;
    using B5 = Eigen::Matrix<double,5,2>;
    using K5 = Eigen::Matrix<double,2,5>;

    std::vector<K5> K(N);

    A5 P = L_aug_;

    for (int i = N - 1; i >= 0; --i)
    {
        A5 Aaug = A5::Zero();
        B5 Baug = B5::Zero();

        // build augmented system (Method A)
        Aaug.block<3,3>(0,0) = As[i];
        Aaug.block<3,2>(0,3) = Bs[i];
        Aaug.block<2,2>(3,3) = Eigen::Matrix2d::Identity();

        Baug.block<3,2>(0,0) = Bs[i];
        Baug.block<2,2>(3,0) = Eigen::Matrix2d::Identity();

        Eigen::Matrix2d S = Rdu_ + Baug.transpose() * P * Baug;

        K[i] = S.ldlt().solve(Baug.transpose() * P * Aaug);

        A5 Pnext =
            Q_aug_
            + Aaug.transpose() * P * Aaug
            - Aaug.transpose() * P * Baug * K[i];

        P = 0.5 * (Pnext + Pnext.transpose());
    }

    return K;
}

// ================= SOLVE =================
Eigen::Vector2d LQRController::solve(
    const Eigen::Vector3d & z0,
    const std::vector<Eigen::Vector3d> & z_ref,
    const std::vector<Eigen::Vector2d> & u_ref)
{
    int N = std::min<int>({horizon_, (int)z_ref.size(), (int)u_ref.size()});
    if (N == 0) return {0,0};

    std::vector<Eigen::Matrix3d> A(N);
    std::vector<Eigen::Matrix<double,3,2>> B(N);

    for (int i = 0; i < N; i++)
        linearize(z_ref[i], u_ref[i], dt_, A[i], B[i]);

    auto K = computeGains(A, B);

    Eigen::Vector3d z = z0;
    Eigen::Vector2d delta_u_prev = delta_u_prev_;

    Eigen::Vector2d u_out = Eigen::Vector2d::Zero();

    for (int i = 0; i < N; i++)
    {
        Eigen::Vector3d e = z - z_ref[i];
        e(2) = wrapAngle(e(2));

        Eigen::Matrix<double,5,1> xi;
        xi << e, delta_u_prev;

        Eigen::Vector2d du = -K[i] * xi;

        Eigen::Vector2d u = delta_u_prev + du;

        u(0) = std::clamp(u(0), v_min_, v_max_);
        u(1) = std::clamp(u(1), w_min_, w_max_);

        delta_u_prev = u;

        if (i == 0) u_out = u;

        z(0) += dt_ * u(0) * std::cos(z(2));
        z(1) += dt_ * u(0) * std::sin(z(2));
        z(2)  = wrapAngle(z(2) + dt_ * u(1));
    }

    delta_u_prev_ = delta_u_prev;
    return u_out;
}

// ================= REF =================
void LQRController::buildReference(
    const nav_msgs::msg::Path & path,
    const Eigen::Vector3d & z0,
    std::vector<Eigen::Vector3d> & z_ref,
    std::vector<Eigen::Vector2d> & u_ref) const
{
    size_t start = 0;
    double best = 1e9;

    for (size_t i = 0; i < path.poses.size(); i++)
    {
        double dx = z0(0) - path.poses[i].pose.position.x;
        double dy = z0(1) - path.poses[i].pose.position.y;
        double d = dx*dx + dy*dy;

        if (d < best) { best = d; start = i; }
    }

    size_t N = std::min<size_t>(horizon_+1, path.poses.size()-start);

    z_ref.clear();
    u_ref.clear();

    for (size_t i = 0; i < N; i++)
    {
        auto & p = path.poses[start+i].pose;

        Eigen::Vector3d z;
        z << p.position.x, p.position.y,
        tf2::getYaw(p.orientation);

        z_ref.push_back(z);

        u_ref.push_back({0.2, 0.0}); // simple stable baseline
    }
}

// ================= NAV2 =================
geometry_msgs::msg::TwistStamped LQRController::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist &,
    nav2_core::GoalChecker *)
{
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header = pose.header;

    if (path_.poses.empty()) return cmd;

    Eigen::Vector3d z0;
    z0 << pose.pose.position.x,
            pose.pose.position.y,
            tf2::getYaw(pose.pose.orientation);

    std::vector<Eigen::Vector3d> z_ref;
    std::vector<Eigen::Vector2d> u_ref;

    buildReference(path_, z0, z_ref, u_ref);

    auto u = solve(z0, z_ref, u_ref);

    cmd.twist.linear.x = u(0);
    cmd.twist.angular.z = u(1);

    return cmd;
}

void LQRController::setSpeedLimit(
    const double & speed_limit, 
    const bool & percentage)
{
    if (percentage) max_linear_vel_ *= speed_limit / 100.0;
    else max_linear_vel_ = speed_limit;
}

} // namespace lqr_nav2_controller

PLUGINLIB_EXPORT_CLASS(
    lqr_nav2_controller::LQRController,
    nav2_core::Controller)