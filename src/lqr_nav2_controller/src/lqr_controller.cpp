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

    node->declare_parameter(name + ".cost_x",     5.0);
    node->declare_parameter(name + ".cost_y",     5.0);
    node->declare_parameter(name + ".cost_theta", 1.0);
    node->declare_parameter(name + ".cost_v",     0.3);
    node->declare_parameter(name + ".cost_w",     0.3);
    node->declare_parameter(name + ".cost_dv",    1.0);
    node->declare_parameter(name + ".cost_dw",    1.0);
    node->declare_parameter(name + ".horizon",     25);
    node->declare_parameter(name + ".dt",         0.1);
    node->declare_parameter(name + ".v_min",     -0.2);
    node->declare_parameter(name + ".v_max",      1.0);
    node->declare_parameter(name + ".w_min",     -1.2);
    node->declare_parameter(name + ".w_max",      1.2);

    double cx  = node->get_parameter(name + ".cost_x").as_double();
    double cy  = node->get_parameter(name + ".cost_y").as_double();
    double cth = node->get_parameter(name + ".cost_theta").as_double();
    double cv  = node->get_parameter(name + ".cost_v").as_double();
    double cw  = node->get_parameter(name + ".cost_w").as_double();
    double cdv = node->get_parameter(name + ".cost_dv").as_double();
    double cdw = node->get_parameter(name + ".cost_dw").as_double();

    horizon_ = node->get_parameter(name + ".horizon").as_int();
    dt_      = node->get_parameter(name + ".dt").as_double();
    v_min_   = node->get_parameter(name + ".v_min").as_double();
    v_max_   = node->get_parameter(name + ".v_max").as_double();
    w_min_   = node->get_parameter(name + ".w_min").as_double();
    w_max_   = node->get_parameter(name + ".w_max").as_double();

    // 3x3 state cost (x, y, theta)
    Qz_ = Eigen::Vector3d(cx, cy, cth).asDiagonal();
    Lz_ = Qz_;

    // 2x2 absolute-velocity cost
    Ru_  = Eigen::Vector2d(cv, cw).asDiagonal();

    // 2x2 control-rate cost
    Rdu_ = Eigen::Vector2d(cdv, cdw).asDiagonal();

    // Augmented 5x5 stage cost  Q_aug = blkdiag(Qz, Ru)
    // Python: self.Q_aug = np.block([[Qx, 0],[0, Ru]])
    Q_aug_.setZero();
    Q_aug_.block<3,3>(0,0) = Qz_;
    Q_aug_.block<2,2>(3,3) = Ru_;

    // Terminal cost — same structure as stage cost
    // Python: self.L_aug = np.block([[Lx, 0],[0, Ru]])
    L_aug_ = Q_aug_;

    u_prev_.setZero();
}

void LQRController::cleanup() {}
void LQRController::activate() {}
void LQRController::deactivate() {}

void LQRController::setPlan(const nav_msgs::msg::Path & path)
{
    path_ = path;
}

// ================= LINEARIZATION =================
// Python: DubinsCar3D2Ctrls.linearize (discrete=True)
void LQRController::linearize(
    const Eigen::Vector3d & z,
    const Eigen::Vector2d & u,
    double dt,
    Eigen::Matrix3d & A,
    Eigen::Matrix<double,3,2> & B) const
    {
    double th = z(2);
    double v = u(0);

    // Continuous-time Jacobians
    Eigen::Matrix3d Ac = Eigen::Matrix3d::Zero();
    Ac(0,2) = -v * std::sin(th);
    Ac(1,2) =  v * std::cos(th);

    Eigen::Matrix<double,3,2> Bc = Eigen::Matrix<double,3,2>::Zero();
    Bc(0,0) = std::cos(th);
    Bc(1,0) = std::sin(th);
    Bc(2,1) = 1.0;

    // Euler discretisation: A = I + dt*Ac,  B = dt*Bc
    A = Eigen::Matrix3d::Identity() + dt * Ac;
    B = dt * Bc;
}

// ================= AUGMENTED DYNAMICS =================
// Python: build_augmented_dynamics
//
//   x_aug = [z (3), u_prev (2)]      (5-vector)
//   new control: du = u - u_prev     (2-vector)
//
//   A_aug = [ A   B ]     B_aug = [ B ]
//           [ 0   I ]             [ I ]
//
void LQRController::buildAugmentedDynamics(
    const std::vector<Eigen::Matrix3d>            & As,
    const std::vector<Eigen::Matrix<double,3,2>>  & Bs,
    std::vector<Eigen::Matrix<double,5,5>>         & A_aug,
    std::vector<Eigen::Matrix<double,5,2>>         & B_aug) const
{
    int N = static_cast<int>(As.size());
    A_aug.resize(N);
    B_aug.resize(N);

    for (int i = 0; i < N; ++i)
    {
        A_aug[i].setZero();
        A_aug[i].block<3,3>(0,0) = As[i];
        A_aug[i].block<3,2>(0,3) = Bs[i];
        A_aug[i].block<2,2>(3,3) = Eigen::Matrix2d::Identity();

        B_aug[i].setZero();
        B_aug[i].block<3,2>(0,0) = Bs[i];
        B_aug[i].block<2,2>(3,0) = Eigen::Matrix2d::Identity();
    }
}

// ================= RICCATI BACKWARD PASS =================
// Python: compute_gains
//
//   S      = Rdu + Bt' * P_{t+1} * Bt
//   K_t    = S^{-1} * Bt' * P_{t+1} * At
//   P_t    = Q_aug + At' * P_{t+1} * At - At' * P_{t+1} * Bt * K_t
//
std::vector<Eigen::Matrix<double,2,5>> LQRController::computeGains(
    const std::vector<Eigen::Matrix<double,5,5>> & A_aug,
    const std::vector<Eigen::Matrix<double,5,2>> & B_aug) const
{
    int N = static_cast<int>(A_aug.size());

    std::vector<Eigen::Matrix<double,2,5>> Ks(N);

    // Terminal cost-to-go
    Eigen::Matrix<double,5,5> P = L_aug_;

    for (int i = N - 1; i >= 0; --i)
    {
        const auto & At = A_aug[i];
        const auto & Bt = B_aug[i];

        Eigen::Matrix2d S = Rdu_ + Bt.transpose() * P * Bt;
        Ks[i] = S.ldlt().solve(Bt.transpose() * P * At);

        // P_t = Q_aug + At'*P*At - At'*P*Bt*K_t
        // No manual symmetrisation — Python exactly
        P = Q_aug_
            + At.transpose() * P * At
            - At.transpose() * P * Bt * Ks[i];
    }

    return Ks;
}

// ================= FORWARD SIMULATION (SOLVE) =================
// Python: LQRSmoothingAlgorithm.solve
Eigen::Vector2d LQRController::solve(
    const Eigen::Vector3d              & z0,
    const std::vector<Eigen::Vector3d> & z_ref,
    const std::vector<Eigen::Vector2d> & u_ref,
    const Eigen::Vector2d              & u_prev_ref_0)
{
    int N = std::min<int>({horizon_,
                           static_cast<int>(z_ref.size()),
                           static_cast<int>(u_ref.size())});
    if (N == 0) return Eigen::Vector2d::Zero();

    // --- Step 1: linearise along reference ---
    std::vector<Eigen::Matrix3d>           As(N);
    std::vector<Eigen::Matrix<double,3,2>> Bs(N);
    for (int i = 0; i < N; ++i)
        linearize(z_ref[i], u_ref[i], dt_, As[i], Bs[i]);

    // --- Step 2: augmented dynamics ---
    std::vector<Eigen::Matrix<double,5,5>> A_aug;
    std::vector<Eigen::Matrix<double,5,2>> B_aug;
    buildAugmentedDynamics(As, Bs, A_aug, B_aug);

    // --- Step 3: backward Riccati pass ---
    auto Ks = computeGains(A_aug, B_aug);

    // --- Precompute reference u_prev and du ---
    // Python forward loop before step 4
    std::vector<Eigen::Vector2d> u_prev_ref(N);
    std::vector<Eigen::Vector2d> du_ref(N);

    u_prev_ref[0] = u_prev_ref_0;
    du_ref[0]     = u_ref[0] - u_prev_ref[0];

    for (int i = 1; i < N; ++i)
    {
        u_prev_ref[i] = u_ref[i - 1];
        du_ref[i]     = u_ref[i] - u_prev_ref[i];
    }

    // --- Step 4: forward simulation ---
    Eigen::Vector3d z       = z0;
    Eigen::Vector2d u_prev  = u_prev_;         // robot's last applied command
    Eigen::Vector2d u_out   = Eigen::Vector2d::Zero();

    for (int i = 0; i < N; ++i)
    {
        // Augmented state error
        Eigen::Matrix<double,5,1> x_aug_now;
        x_aug_now << z, u_prev;

        Eigen::Matrix<double,5,1> x_aug_ref;
        x_aug_ref << z_ref[i], u_prev_ref[i];

        Eigen::Matrix<double,5,1> delta_aug = x_aug_now - x_aug_ref;

        // Wrap heading error — Python arctan2 wrap
        delta_aug(2) = wrapAngle(delta_aug(2));

        // Closed-loop control: du = du_ref - K * delta_aug
        Eigen::Vector2d du = du_ref[i] - Ks[i] * delta_aug;
        Eigen::Vector2d u  = u_prev + du;

        u(0) = std::clamp(u(0), v_min_, v_max_);
        u(1) = std::clamp(u(1), w_min_, w_max_);

        if (i == 0) u_out = u;

        // Propagate nonlinear Dubins dynamics (Python dynsys.forward_np)
        z(0) += dt_ * u(0) * std::cos(z(2));
        z(1) += dt_ * u(0) * std::sin(z(2));
        z(2)  = wrapAngle(z(2) + dt_ * u(1));

        u_prev = u;
    }

    u_prev_ = u_prev;
    return u_out;
}

// ================= BUILD REFERENCE =================
// Python: closest_reference_index + sample_reference_window
// + the u_prev_ref_0 logic in get_action
void LQRController::buildReference(
    const nav_msgs::msg::Path  & path,
    const Eigen::Vector3d      & z0,
    std::vector<Eigen::Vector3d> & z_ref,
    std::vector<Eigen::Vector2d> & u_ref,
    Eigen::Vector2d              & u_prev_ref_0) const
{
    // --- Find closest waypoint (Python closest_reference_index) ---
    size_t start = 0;
    double best  = std::numeric_limits<double>::max();

    for (size_t i = 0; i < path.poses.size(); ++i)
    {
        double dx = z0(0) - path.poses[i].pose.position.x;
        double dy = z0(1) - path.poses[i].pose.position.y;
        double d  = dx*dx + dy*dy;
        if (d < best) { best = d; start = i; }
    }

    // u_prev_ref_0: if not at the very beginning, look one step back
    // Python:
    //   if ref_idx > 0: u_prev_ref_0 = u_ref[ref_idx - 1]
    //   else:           u_prev_ref_0 = zeros
    if (start > 0)
    {
        // We need the velocity implied by the step before 'start'.
        // Compute it from consecutive path poses.
        auto & p0 = path.poses[start - 1].pose;
        auto & p1 = path.poses[start].pose;
        double dx  = p1.position.x - p0.position.x;
        double dy  = p1.position.y - p0.position.y;
        double v   = std::sqrt(dx*dx + dy*dy) / dt_;
        double dth = wrapAngle(tf2::getYaw(p1.orientation) - tf2::getYaw(p0.orientation));
        double w   = dth / dt_;
        u_prev_ref_0 = {v, w};
    }
    else
    {
        u_prev_ref_0.setZero();
    }

    // --- Extract horizon-length window (Python sample_reference_window) ---
    size_t n_available = path.poses.size() - start;
    // +1 so we can derive u_ref for all N steps (need N+1 poses)
    size_t N = std::min<size_t>(static_cast<size_t>(horizon_) + 1, n_available);

    z_ref.clear();
    z_ref.reserve(N);

    for (size_t i = 0; i < N; ++i)
    {
        auto & p = path.poses[start + i].pose;
        Eigen::Vector3d z;
        z << p.position.x, p.position.y, tf2::getYaw(p.orientation);
        z_ref.push_back(z);
    }

    // --- Derive u_ref from consecutive z_ref poses ---
    // Python buildReference velocity computation:
    //   v = sqrt(dx^2+dy^2) / dt,  w = dtheta / dt
    u_ref.clear();
    u_ref.reserve(z_ref.size());

    for (size_t i = 1; i < z_ref.size(); ++i)
    {
        double dx  = z_ref[i](0) - z_ref[i-1](0);
        double dy  = z_ref[i](1) - z_ref[i-1](1);
        double v   = std::sqrt(dx*dx + dy*dy) / dt_;
        double dth = wrapAngle(z_ref[i](2) - z_ref[i-1](2));
        double w   = dth / dt_;
        u_ref.push_back({v, w});
    }

    // Keep z_ref and u_ref the same length (drop the extra pose used only for u)
    if (!u_ref.empty() && z_ref.size() > u_ref.size())
        z_ref.resize(u_ref.size());

    // If there are fewer steps than the horizon, zero-pad u_ref
    // Python: u_ref_win[n_remaining:] = 0.0
    while (static_cast<int>(u_ref.size()) < horizon_)
        u_ref.push_back(Eigen::Vector2d::Zero());

    while (static_cast<int>(z_ref.size()) < horizon_)
        z_ref.push_back(z_ref.empty() ? z0 : z_ref.back());
}

// ================= NAV2 ENTRY POINT =================
geometry_msgs::msg::TwistStamped LQRController::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist       & velocity,
    nav2_core::GoalChecker *)
{
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header = pose.header;

    if (path_.poses.empty()) return cmd;

    Eigen::Vector3d z0;
    z0 << pose.pose.position.x,
          pose.pose.position.y,
          tf2::getYaw(pose.pose.orientation);

    // Seed u_prev_ from odometry on the very first call (Python prev_u init)
    // After the first step u_prev_ is maintained internally by solve().
    static bool first_call = true;
    if (first_call)
    {
        u_prev_(0) = velocity.linear.x;
        u_prev_(1) = velocity.angular.z;
        first_call  = false;
    }

    std::vector<Eigen::Vector3d> z_ref;
    std::vector<Eigen::Vector2d> u_ref;
    Eigen::Vector2d              u_prev_ref_0;

    buildReference(path_, z0, z_ref, u_ref, u_prev_ref_0);

    auto u = solve(z0, z_ref, u_ref, u_prev_ref_0);

    cmd.twist.linear.x  = u(0);
    cmd.twist.angular.z = u(1);

    return cmd;
}

void LQRController::setSpeedLimit(
    const double & speed_limit,
    const bool   & percentage)
{
    // Apply limit to v_max_ (which is used in the clamp inside solve)
    if (percentage)
        v_max_ = v_max_ * speed_limit / 100.0;
    else
        v_max_ = speed_limit;
}

} // namespace lqr_nav2_controller

PLUGINLIB_EXPORT_CLASS(
    lqr_nav2_controller::LQRController,
    nav2_core::Controller)