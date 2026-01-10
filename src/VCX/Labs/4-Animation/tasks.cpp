#include "Labs/4-Animation/tasks.h"
#include "CustomFunc.inl"
#include "IKSystem.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <spdlog/spdlog.h>

namespace VCX::Labs::Animation {
    void ForwardKinematics(IKSystem & ik, int StartIndex) {
        if (StartIndex == 0) {
            ik.JointGlobalRotation[0] = ik.JointLocalRotation[0];
            ik.JointGlobalPosition[0] = ik.JointLocalOffset[0];
            StartIndex                = 1;
        }

        for (int i = StartIndex; i < ik.JointLocalOffset.size(); i++) {
            // your code here: forward kinematics, update JointGlobalPosition and JointGlobalRotation
            ik.JointGlobalRotation[i] = ik.JointGlobalRotation[i - 1] * ik.JointLocalRotation[i];
            ik.JointGlobalPosition[i] = ik.JointGlobalPosition[i - 1] + ik.JointGlobalRotation[i - 1] * ik.JointLocalOffset[i];
        }
    }

    void InverseKinematicsCCD(IKSystem & ik, const glm::vec3 & EndPosition, int maxCCDIKIteration, float eps) {
        ForwardKinematics(ik, 0);
        // These functions will be useful: glm::normalize, glm::rotation, glm::quat * glm::quat
        int CCDIKIteration = 0;
        for (CCDIKIteration = 0; CCDIKIteration < maxCCDIKIteration && glm::l2Norm(ik.EndEffectorPosition() - EndPosition) > eps; CCDIKIteration++) {
            // your code here: ccd ik
            for (int joint_idx = ik.NumJoints() - 2; joint_idx >= 0; joint_idx--) {
                glm::vec3 to_end_effector = ik.EndEffectorPosition() - ik.JointGlobalPosition[joint_idx];
                glm::vec3 to_target       = EndPosition - ik.JointGlobalPosition[joint_idx];

                glm::quat rotation_delta         = glm::rotation(glm::normalize(to_end_effector), glm::normalize(to_target));
                ik.JointLocalRotation[joint_idx] = rotation_delta * ik.JointLocalRotation[joint_idx];

                ForwardKinematics(ik, joint_idx);
            }
        }
        // std::cout << "CCDIKIteration: " << CCDIKIteration << std::endl;
    }

    void InverseKinematicsFABR(IKSystem & ik, const glm::vec3 & EndPosition, int maxFABRIKIteration, float eps) {
        ForwardKinematics(ik, 0);
        int                    nJoints = ik.NumJoints();
        std::vector<glm::vec3> backward_positions(nJoints, glm::vec3(0, 0, 0)), forward_positions(nJoints, glm::vec3(0, 0, 0));
        int                    IKIteration = 0;
        for (IKIteration = 0; IKIteration < maxFABRIKIteration && glm::l2Norm(ik.EndEffectorPosition() - EndPosition) > eps; IKIteration++) {
            // task: fabr ik
            // backward update
            glm::vec3 next_position         = EndPosition;
            backward_positions[nJoints - 1] = EndPosition;

            for (int i = nJoints - 2; i >= 0; i--) {
                // your code here
                glm::vec3 dir         = glm::normalize(ik.JointGlobalPosition[i] - next_position);
                backward_positions[i] = next_position + dir * ik.JointOffsetLength[i + 1];
                next_position         = backward_positions[i];
            }

            // forward update
            glm::vec3 now_position = ik.JointGlobalPosition[0];
            forward_positions[0]   = ik.JointGlobalPosition[0];
            for (int i = 0; i < nJoints - 1; i++) {
                // your code here
                glm::vec3 dir            = glm::normalize(backward_positions[i + 1] - now_position);
                forward_positions[i + 1] = now_position + dir * ik.JointOffsetLength[i + 1];
                now_position             = forward_positions[i + 1];
            }
            ik.JointGlobalPosition = forward_positions; // copy forward positions to joint_positions
        }

        // Compute joint rotation by position here.
        for (int i = 0; i < nJoints - 1; i++) {
            ik.JointGlobalRotation[i] = glm::rotation(glm::normalize(ik.JointLocalOffset[i + 1]), glm::normalize(ik.JointGlobalPosition[i + 1] - ik.JointGlobalPosition[i]));
        }
        ik.JointLocalRotation[0] = ik.JointGlobalRotation[0];
        for (int i = 1; i < nJoints - 1; i++) {
            ik.JointLocalRotation[i] = glm::inverse(ik.JointGlobalRotation[i - 1]) * ik.JointGlobalRotation[i];
        }
        ForwardKinematics(ik, 0);
        // std::cout << "FABRIKIteration: " << IKIteration << std::endl;
    }

    IKSystem::Vec3ArrPtr IKSystem::BuildCustomTargetPosition() {
        // get function from https://www.wolframalpha.com/input/?i=Albert+Einstein+curve
        int nums      = 5000;
        using Vec3Arr = std::vector<glm::vec3>;
        std::shared_ptr<Vec3Arr> custom(new Vec3Arr(nums));
        int                      index = 0;
        for (int i = 0; i < nums; i++) {
            float x_val = 1.5e-3f * custom_x(92 * glm::pi<float>() * i / nums);
            float y_val = 1.5e-3f * custom_y(92 * glm::pi<float>() * i / nums);
            if (std::abs(x_val) < 1e-3 || std::abs(y_val) < 1e-3) continue;
            (*custom)[index++] = glm::vec3(0.5f - x_val, 0.0f, y_val + 0.2f);
        }
        custom->resize(index);
        return custom;
    }

    static Eigen::VectorXf glm2eigen(std::vector<glm::vec3> const & glm_v) {
        Eigen::VectorXf v = Eigen::Map<Eigen::VectorXf const, Eigen::Aligned>(reinterpret_cast<float const *>(glm_v.data()), static_cast<int>(glm_v.size() * 3));
        return v;
    }

    static std::vector<glm::vec3> eigen2glm(Eigen::VectorXf const & eigen_v) {
        return std::vector<glm::vec3>(
            reinterpret_cast<glm::vec3 const *>(eigen_v.data()),
            reinterpret_cast<glm::vec3 const *>(eigen_v.data() + eigen_v.size()));
    }

    static Eigen::SparseMatrix<float> CreateEigenSparseMatrix(std::size_t n, std::vector<Eigen::Triplet<float>> const & triplets) {
        Eigen::SparseMatrix<float> matLinearized(n, n);
        matLinearized.setFromTriplets(triplets.begin(), triplets.end());
        return matLinearized;
    }

    // solve Ax = b and return x
    static Eigen::VectorXf ComputeSimplicialLLT(
        Eigen::SparseMatrix<float> const & A,
        Eigen::VectorXf const &            b) {
        auto solver = Eigen::SimplicialLLT<Eigen::SparseMatrix<float>>(A);
        return solver.solve(b);
    }

    void AdvanceMassSpringSystem(MassSpringSystem & system, float const dt) {
        // your code here: rewrite following code
        // 隐式欧拉法 （M+dt*dt*H) * \delta_x = -(M(x-y)+dt*dt*\delta_Energy); y = x + dt*v + dt*dt*gravity
        std::size_t                        n = system.Positions.size();
        std::vector<Eigen::Triplet<float>> triplets;
        Eigen::VectorXf                    b = Eigen::VectorXf::Zero(static_cast<int>(n * 3));
        // 我们首先构建b,以及A中的质量项
        for (std::size_t i = 0; i < n; i++) {
            if (system.Fixed[i]) {
                // 固定点方程: 1 * dx = 0
                triplets.emplace_back(i * 3 + 0, i * 3 + 0, 1.0f);
                triplets.emplace_back(i * 3 + 1, i * 3 + 1, 1.0f);
                triplets.emplace_back(i * 3 + 2, i * 3 + 2, 1.0f);
                continue; // b 保持为 0
            }
            for (int k = 0; k < 3; k++) {
                triplets.emplace_back(static_cast<int>(i * 3 + k), static_cast<int>(i * 3 + k), system.Mass);
                b(static_cast<int>(i * 3 + k)) = system.Mass * (dt * system.Velocities[i][k] + dt * dt * ((k == 1) ? -system.Gravity : 0.0f));
            }
        }
        // 然后我们计算弹簧的能量对位置的一阶导数和二阶导数
        for (auto const & spring : system.Springs) {
            std::size_t i0 = spring.AdjIdx.first, i1 = spring.AdjIdx.second;
            glm::vec3   p0 = system.Positions[i0], p1 = system.Positions[i1];
            // 如果两个端点都被固定则跳过
            if (system.Fixed[i0] && system.Fixed[i1]) continue;
            // 我们取方向向量为从p0指向p1
            glm::vec3 dir = p1 - p0;
            float     len = glm::length(dir);
            if (len < 1e-6f) continue;
            // 这里我们假设弹簧方向不为0，且弹簧方向dt内不变
            dir = glm::normalize(dir);
            // 这里弹簧力的方向是p0到p1收缩的方向，所以对p0是正的，对p1是负的
            float     f_mag      = system.Stiffness * (len - spring.RestLength);
            glm::vec3 f_spring_i = f_mag * dir;

            if (! system.Fixed[i0]) {
                b(i0 * 3 + 0) += dt * dt * f_spring_i.x;
                b(i0 * 3 + 1) += dt * dt * f_spring_i.y;
                b(i0 * 3 + 2) += dt * dt * f_spring_i.z;
            }
            if (! system.Fixed[i1]) {
                b(i1 * 3 + 0) -= dt * dt * f_spring_i.x;
                b(i1 * 3 + 1) -= dt * dt * f_spring_i.y;
                b(i1 * 3 + 2) -= dt * dt * f_spring_i.z;
            }

            // 二阶导数贡献到A上，计算Hessian矩阵
            double HMatrix[3][3] = { 0 };
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    // Hii = -k * [ (1 - L/l) * I + (L/l) * n * n^T ]
                    HMatrix[row][col] = system.Stiffness * ((row == col ? 1.0f : 0.0f) - spring.RestLength / len * ((row == col ? 1.0f : 0.0f) - dir[row] * dir[col]));
                }
            }
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    if (! system.Fixed[i0]) {
                        triplets.emplace_back(static_cast<int>(i0 * 3 + row), static_cast<int>(i0 * 3 + col), HMatrix[row][col] * dt * dt);
                        triplets.emplace_back(static_cast<int>(i0 * 3 + row), static_cast<int>(i1 * 3 + col), -HMatrix[row][col] * dt * dt);
                    }
                    if (! system.Fixed[i1]) {
                        triplets.emplace_back(static_cast<int>(i1 * 3 + row), static_cast<int>(i0 * 3 + col), -HMatrix[row][col] * dt * dt);
                        triplets.emplace_back(static_cast<int>(i1 * 3 + row), static_cast<int>(i1 * 3 + col), HMatrix[row][col] * dt * dt);
                    }
                }
            }
        }
        Eigen::SparseMatrix<float> A       = CreateEigenSparseMatrix(n * 3, triplets);
        Eigen::VectorXf            delta_x = ComputeSimplicialLLT(A, b);
        Eigen::VectorXf            x       = glm2eigen(system.Positions);
        x += delta_x;
        system.Positions = eigen2glm(x);
        // update velocity
        for (std::size_t i = 0; i < n; i++) {
            if (system.Fixed[i]) continue;
            system.Velocities[i] = glm::vec3(
                delta_x(static_cast<int>(i * 3 + 0)) / dt,
                delta_x(static_cast<int>(i * 3 + 1)) / dt,
                delta_x(static_cast<int>(i * 3 + 2)) / dt);
        }
    }
} // namespace VCX::Labs::Animation
