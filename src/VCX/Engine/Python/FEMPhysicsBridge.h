#pragma once

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include "Engine/Mesh/MeshDataLoader.h"
#include "Engine/Mesh/VisualizationBuilder.h"

namespace py = pybind11;

namespace VCX::Engine::Python::FEM {

// 可视化数据结构
struct MeshVisualizationData {
    struct Body {
        std::vector<glm::vec3> vertices;
        std::vector<glm::uvec3> faces;  // 三角形索引
        std::vector<glm::vec3> colors;  // 每个面的颜色（可选）
    };
    
    std::vector<Body> bodies;
};

// 外力控制结构
struct ExternalForce {
    std::vector<bool> force_verts_mask;
    bool hasExternalForces = true;
    bool pull_X;
    bool pull_Y;
    bool pull_Z;
    bool has_X;
    bool has_Y;
    bool has_Z;
    double pull_strength;
    std::pair<double,double> pull_strength_minmax;
};

class PhysicsBridge {
public:
    PhysicsBridge(const std::string& pythonPath, 
                  const std::string& systemName,
                  const std::string& problemName,
                  const std::string& subspaceModelPath);
    ~PhysicsBridge();
    
    // 状态管理
    // 重新设置物理状态
    void ResetState();
    // 停止速度
    void StopVelocity();
    // 获取当前状态
    std::vector<double> GetState() const;
    // 设置当前状态
    void SetState(const std::vector<double>& state);
    
    // 仿真
    void Timestep(float dt);
    
    // 查询
    int GetFullDim() const;
    int GetSubspaceDim() const;
    bool UseSubspace() const;
    double GetPotentialEnergy();
    
    // 可视化
    MeshVisualizationData GetVisualizationData();
    
    // 外力控制
    ExternalForce GetExternalForce() const { return _externalForce; }
    void SetExternalForce(const std::string& axis, bool value);
    void SetForceStrength(float strength);
    
    // 重新加载问题
    void ReloadProblem(const std::string& problemName, const std::string& subspaceModelPath = "");
    
    // 子空间模型管理
    void SetSubspaceModel(const std::string& modelPath);
    
private:
    py::object _system;             // Python FEMSystem对象
    py::object _system_def;         // system_def字典
    py::object _int_state;          // 积分器状态
    py::object _int_opts;           // 积分器选项
    
    bool _useSubspace;
    int _fullDim;
    int _subspaceDim;
    int _posDim;  // 2D or 3D
    ExternalForce _externalForce;
    
    // 子空间相关
    py::object _subspace_apply;     // 子空间映射函数
    py::object _subspace_domain_dict; // 子空间域信息
    double _t_schedule_final;       // 扩散调度最终值
    
    // C++网格数据（无需Python）
    Mesh::FEMMeshData _meshData;
    
    // 子空间映射
    py::object StateToSystem(const py::object& state);
    // 初始化Python环境和路径
    void InitializePython(const std::string& pythonPath);
    // 加载物理系统
    void LoadSystem(const std::string& systemName, 
                   const std::string& problemName,
                   const std::string& subspaceModelPath);
};

} // namespace VCX::Engine::Python::FEM
