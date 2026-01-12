#include "Engine/Python/FEMPhysicsBridge.h"
#include "Engine/Python/PythonInterpreter.h"
#include <spdlog/spdlog.h>
#include <iostream>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;

namespace VCX::Engine::Python::FEM {

PhysicsBridge::PhysicsBridge(const std::string& pythonPath,
                             const std::string& systemName,
                             const std::string& problemName,
                             const std::string& subspaceModelPath) 
    : _system(), _system_def(), _int_state(), _int_opts(),
      _useSubspace(false), _fullDim(0), _subspaceDim(0), _posDim(0) {
    
    try {
        // 确保全局 Python 解释器已初始化（单例模式）
        PythonInterpreter::GetInstance().EnsureInitialized();
        InitializePython(pythonPath);
        LoadSystem(systemName, problemName, subspaceModelPath);
        spdlog::info("[PhysicsBridge] Initialized: {} - {}", systemName, problemName);
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] Initialization failed: {}", e.what());
        throw;
    }
}

PhysicsBridge::~PhysicsBridge() {
    spdlog::info("[PhysicsBridge] Ending PhysicsBridge");
}

void PhysicsBridge::InitializePython(const std::string& pythonPath) {
    // 添加Python路径
    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    
    // 添加conda环境路径
    if (!pythonPath.empty()) {
        path.insert(0, pythonPath);
    }
    
    // 添加项目python目录（相对于当前工作目录），以便导入python模块
    path.insert(0, "./python");
    
    // 打印当前sys.path用于调试
    spdlog::info("[PhysicsBridge] Python sys.path:");
    for (auto item : path) {
        spdlog::info("  - {}", py::str(item).cast<std::string>());
    }
}

// ================ 加载物理系统（相关函数可以见fem_modal.py） ====================
void PhysicsBridge::LoadSystem(const std::string& systemName,
                               const std::string& problemName,
                               const std::string& subspaceModelPath) {
    
    // 导入必要的Python模块
    py::module_ fem_model = py::module_::import("fem_model");
    py::module_ integrators = py::module_::import("integrators");
    py::module_ np = py::module_::import("numpy");
    py::module_ jnp = py::module_::import("jax.numpy");
    
    // 构建FEM系统
    py::object FEMSystem = fem_model.attr("FEMSystem");
    py::tuple result = FEMSystem.attr("construct")(problemName);
    _system = result[0];
    _system_def = result[1];
    
    // 获取维度信息
    _fullDim = py::cast<int>(_system_def["init_pos"].attr("size"));
    _posDim = py::cast<int>(_system.attr("pos_dim"));
    
    // 检查是否使用子空间
    _useSubspace = !subspaceModelPath.empty();
    
    if (_useSubspace) {
        // TODO: 加载子空间模型
        _subspaceDim = 8;  // 默认值，应该从模型加载
        spdlog::warn("[PhysicsBridge] Subspace mode not fully implemented yet");
        _useSubspace = false;  // 暂时禁用
    } else {
        _subspaceDim = _fullDim;
    }
    
    // 初始化积分器
    _int_opts = py::dict();
    _int_state = py::dict();
    
    // 使用隐式近端积分器
    integrators.attr("initialize_integrator")(_int_opts, _int_state, "implicit-proximal");
    
    // 设置初始状态
    _int_state["q_t"] = _system_def["init_pos"];
    _int_state["q_tm1"] = _system_def["init_pos"];
    _int_state["qdot_t"] = jnp.attr("zeros_like")(_system_def["init_pos"]);
    py::dict forces = _system_def["external_forces"].cast<py::dict>();
    if (forces.contains("force_verts_mask")) _externalForce.force_verts_mask = forces["force_verts_mask"].cast<std::vector<bool>>();
    if (forces.contains("pull_X")) {
        _externalForce.has_X = true;
        double val = forces["pull_X"].cast<double>();
        _externalForce.pull_X = (val != 0.0);
    } else {
        _externalForce.has_X = false;
        _externalForce.pull_X = false;  // default off when not provided
    }
    if (forces.contains("pull_Y")) {
        _externalForce.has_Y = true;
        double val = forces["pull_Y"].cast<double>();
        _externalForce.pull_Y = (val != 0.0);
    } else {
        _externalForce.has_Y = false;
        _externalForce.pull_Y = false;
    }
    if (forces.contains("pull_Z")) {
        _externalForce.has_Z = true;
        double val = forces["pull_Z"].cast<double>();
        _externalForce.pull_Z = (val != 0.0);
    } else {
        _externalForce.has_Z = false;
        _externalForce.pull_Z = false;
    }
    if (forces.contains("pull_strength")) {
        _externalForce.pull_strength = forces["pull_strength"].cast<double>();
        _externalForce.hasExternalForces = true;
    } else {
        _externalForce.pull_strength = 0.0;
        _externalForce.hasExternalForces = false;
    }
    if (forces.contains("pull_strength_minmax")) {
        _externalForce.pull_strength_minmax = forces["pull_strength_minmax"].cast<std::pair<double,double>>();
    } else {
        _externalForce.pull_strength_minmax = std::make_pair(0.0, 1.0);
    }
    spdlog::info("[PhysicsBridge] System loaded: dim={}, pos_dim={}", _fullDim, _posDim);
}

void PhysicsBridge::ReloadProblem(const std::string& problemName) {
    spdlog::info("[PhysicsBridge] Reloading problem: {}", problemName);
    
    try {
        // 重新加载系统，不需要重新初始化Python
        LoadSystem("fem", problemName, "");
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] ReloadProblem failed: {}", e.what());
        throw;
    }
}

py::object PhysicsBridge::StateToSystem(const py::object& state) {
    if (_useSubspace) {
        // TODO: 应用子空间映射
        return state;
    } else {
        return state;
    }
}

void PhysicsBridge::ResetState() {
    py::module_ jnp = py::module_::import("jax.numpy");
    _int_state["q_t"] = _system_def["init_pos"];
    _int_state["q_tm1"] = _system_def["init_pos"];
    _int_state["qdot_t"] = jnp.attr("zeros_like")(_system_def["init_pos"]);
    py::dict forces = _system_def["external_forces"].cast<py::dict>();
    if (forces.contains("force_verts_mask")) _externalForce.force_verts_mask = forces["force_verts_mask"].cast<std::vector<bool>>();
    if (forces.contains("pull_X")) {
        _externalForce.has_X = true;
        double val = forces["pull_X"].cast<double>();
        _externalForce.pull_X = (val != 0.0);
    } else {
        _externalForce.has_X = false;
        _externalForce.pull_X = false;
    }
    if (forces.contains("pull_Y")) {
        _externalForce.has_Y = true;
        double val = forces["pull_Y"].cast<double>();
        _externalForce.pull_Y = (val != 0.0);
    } else {
        _externalForce.has_Y = false;
        _externalForce.pull_Y = false;
    }
    if (forces.contains("pull_Z")) {
        _externalForce.has_Z = true;
        double val = forces["pull_Z"].cast<double>();
        _externalForce.pull_Z = (val != 0.0);
    } else {
        _externalForce.has_Z = false;
        _externalForce.pull_Z = false;
    }
    if (forces.contains("pull_strength")) {
        _externalForce.pull_strength = forces["pull_strength"].cast<double>();
    } else {
        _externalForce.pull_strength = 0.0;
    }
    if (forces.contains("pull_strength_minmax")) {
        _externalForce.pull_strength_minmax = forces["pull_strength_minmax"].cast<std::pair<double,double>>();
    } else {
        _externalForce.pull_strength_minmax = std::make_pair(0.0, 1.0);
    }
}

void PhysicsBridge::StopVelocity() {
    py::module_ jnp = py::module_::import("jax.numpy");
    _int_state["qdot_t"] = jnp.attr("zeros_like")(_int_state["q_t"]);
    // 注意不仅要清零速度，还要同步更新上一个位置，使得隐式积分得到的速度为零
    _int_state["q_tm1"] = _int_state["q_t"];
}

std::vector<double> PhysicsBridge::GetState() const {
    py::array_t<double> state_array = py::cast<py::array_t<double>>(_int_state["q_t"]);
    auto buf = state_array.request();
    std::vector<double> state(buf.size);
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < buf.size; i++) {
        state[i] = ptr[i];
    }
    return state;
}

void PhysicsBridge::SetState(const std::vector<double>& state) {
    py::module_ jnp = py::module_::import("jax.numpy");
    py::array_t<double> state_array(state.size(), state.data());
    _int_state["q_t"] = jnp.attr("array")(state_array);
}

void PhysicsBridge::Timestep(float dt) {
    try {
        py::module_ integrators = py::module_::import("integrators");
        
        // 执行时间步
        _int_state = integrators.attr("timestep")(
            _system,
            _system_def,
            _int_state,
            _int_opts,
            "subspace_fn"_a = py::none(),
            "subspace_domain_dict"_a = py::none()
        );
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] Timestep failed: {}", e.what());
    }
}

int PhysicsBridge::GetFullDim() const {
    return _fullDim;
}

int PhysicsBridge::GetSubspaceDim() const {
    return _subspaceDim;
}

bool PhysicsBridge::UseSubspace() const {
    return _useSubspace;
}

double PhysicsBridge::GetPotentialEnergy() {
    try {
        py::object q = StateToSystem(_int_state["q_t"]);
        py::object energy = _system.attr("potential_energy")(_system_def, q);
        return py::cast<double>(energy);
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] GetPotentialEnergy failed: {}", e.what());
        return 0.0;
    }
}

MeshVisualizationData PhysicsBridge::GetVisualizationData() {
    MeshVisualizationData data;
    
    try {
        py::module_ np = py::module_::import("numpy");
        
        // 获取当前状态
        py::object q = StateToSystem(_int_state["q_t"]);
        
        // 使用get_visualization_data方法获取网格数据
        py::dict mesh_data = py::cast<py::dict>(
            _system.attr("get_visualization_data")(_system, _system_def, q)
        );
        
        MeshVisualizationData::Body body;
        
        // 获取顶点数据
        py::array_t<double> verts_array = py::cast<py::array_t<double>>(
            np.attr("array")(mesh_data["vertices"])
        );
        auto verts_buf = verts_array.request();
        int numVerts = verts_buf.shape[0];
        int dim = verts_buf.shape[1];
        double* verts_ptr = static_cast<double*>(verts_buf.ptr);
        
        for (int i = 0; i < numVerts; i++) {
            if (dim == 2) {
                body.vertices.push_back(glm::vec3(
                    static_cast<float>(verts_ptr[i * dim + 0]),
                    static_cast<float>(verts_ptr[i * dim + 1]),
                    0.0f
                ));
            } else {
                body.vertices.push_back(glm::vec3(
                    static_cast<float>(verts_ptr[i * dim + 0]),
                    static_cast<float>(verts_ptr[i * dim + 1]),
                    static_cast<float>(verts_ptr[i * dim + 2])
                ));
            }
        }
        
        // 获取面数据
        py::array_t<int> faces_array = py::cast<py::array_t<int>>(
            np.attr("array")(mesh_data["faces"])
        );
        auto faces_buf = faces_array.request();
        int numFaces = faces_buf.shape[0];
        int* faces_ptr = static_cast<int*>(faces_buf.ptr);
        
        for (int i = 0; i < numFaces; i++) {
            body.faces.push_back(glm::uvec3(
                faces_ptr[i * 3 + 0],
                faces_ptr[i * 3 + 1],
                faces_ptr[i * 3 + 2]
            ));
        }
        
        // 获取regions/材料颜色（如果有）
        if (mesh_data.contains("regions")) {
            py::array_t<int> regions_array = py::cast<py::array_t<int>>(
                np.attr("array")(mesh_data["regions"])
            );
            auto regions_buf = regions_array.request();
            int* regions_ptr = static_cast<int*>(regions_buf.ptr);
            int numRegions = regions_buf.shape[0];
            
            // 检查face_type，判断是否需要映射regions
            std::string face_type = mesh_data.contains("face_type") 
                ? py::cast<std::string>(mesh_data["face_type"]) 
                : "unknown";
            
            // 如果是boundary triangles且regions数量与faces不匹配，需要映射
            if (face_type == "surface" && numRegions != numFaces) {
                // regions是基于tets的，需要为每个boundary face找到对应的tet
                py::object mesh = _system.attr("mesh");
                py::array_t<int> tets_array = py::cast<py::array_t<int>>(
                    np.attr("array")(mesh["E"])
                );
                auto tets_buf = tets_array.request();
                int* tets_ptr = static_cast<int*>(tets_buf.ptr);
                int numTets = tets_buf.shape[0];
                
                // 为每个boundary face找到包含它的tet
                for (int i = 0; i < numFaces; i++) {
                    const auto& face = body.faces[i];
                    std::set<int> face_verts = {(int)face.x, (int)face.y, (int)face.z};
                    
                    int found_region = 0;  // 默认region
                    // 查找包含这三个顶点的tet
                    for (int t = 0; t < numTets; t++) {
                        std::set<int> tet_verts = {
                            tets_ptr[t * 4 + 0],
                            tets_ptr[t * 4 + 1],
                            tets_ptr[t * 4 + 2],
                            tets_ptr[t * 4 + 3]
                        };
                        
                        // 检查face的三个顶点是否都在tet中
                        bool all_in = true;
                        for (int v : face_verts) {
                            if (tet_verts.find(v) == tet_verts.end()) {
                                all_in = false;
                                break;
                            }
                        }
                        
                        if (all_in) {
                            found_region = regions_ptr[t];
                            break;
                        }
                    }
                    
                    // 根据region分配颜色
                    glm::vec3 color;
                    if (found_region == 0) {
                        color = glm::vec3(0.3f, 0.5f, 0.9f);  // 蓝色
                    } else if (found_region == 1) {
                        color = glm::vec3(0.4f, 0.8f, 0.4f);  // 绿色
                    } else if (found_region == 2) {
                        color = glm::vec3(0.9f, 0.4f, 0.3f);  // 红色
                    } else {
                        color = glm::vec3(0.7f, 0.7f, 0.7f);  // 灰色
                    }
                    body.colors.push_back(color);
                }
            } else {
                // regions直接对应faces（2D情况或volume渲染）
                for (int i = 0; i < numFaces; i++) {
                    int region = i < numRegions ? regions_ptr[i] : 0;
                    glm::vec3 color;
                    if (region == 0) {
                        color = glm::vec3(0.3f, 0.5f, 0.9f);  // 蓝色
                    } else if (region == 1) {
                        color = glm::vec3(0.4f, 0.8f, 0.4f);  // 绿色
                    } else if (region == 2) {
                        color = glm::vec3(0.9f, 0.4f, 0.3f);  // 红色
                    } else {
                        color = glm::vec3(0.7f, 0.7f, 0.7f);
                    }
                    body.colors.push_back(color);
                }
            }
        }
        
        data.bodies.push_back(body);
        
        // 处理额外的几何体（比如bistable的endcaps）
        if (mesh_data.contains("extra_vertices") && mesh_data.contains("extra_faces")) {
            MeshVisualizationData::Body extra_body;
            
            // 获取额外的顶点
            py::array_t<double> extra_verts_array = py::cast<py::array_t<double>>(
                np.attr("array")(mesh_data["extra_vertices"])
            );
            auto extra_verts_buf = extra_verts_array.request();
            int numExtraVerts = extra_verts_buf.shape[0];
            int extraDim = extra_verts_buf.shape[1];
            double* extra_verts_ptr = static_cast<double*>(extra_verts_buf.ptr);
            
            for (int i = 0; i < numExtraVerts; i++) {
                if (extraDim == 2) {
                    extra_body.vertices.push_back(glm::vec3(
                        static_cast<float>(extra_verts_ptr[i * extraDim + 0]),
                        static_cast<float>(extra_verts_ptr[i * extraDim + 1]),
                        0.0f
                    ));
                } else {
                    extra_body.vertices.push_back(glm::vec3(
                        static_cast<float>(extra_verts_ptr[i * extraDim + 0]),
                        static_cast<float>(extra_verts_ptr[i * extraDim + 1]),
                        static_cast<float>(extra_verts_ptr[i * extraDim + 2])
                    ));
                }
            }
            
            // 获取额外的面
            py::array_t<int> extra_faces_array = py::cast<py::array_t<int>>(
                np.attr("array")(mesh_data["extra_faces"])
            );
            auto extra_faces_buf = extra_faces_array.request();
            int numExtraFaces = extra_faces_buf.shape[0];
            int vertsPerFace = extra_faces_buf.shape[1];
            int* extra_faces_ptr = static_cast<int*>(extra_faces_buf.ptr);
            
            // 检查是四边形还是三角形
            if (vertsPerFace == 4) {
                // 四边形，分解为两个三角形
                for (int i = 0; i < numExtraFaces; i++) {
                    int v0 = extra_faces_ptr[i * 4 + 0];
                    int v1 = extra_faces_ptr[i * 4 + 1];
                    int v2 = extra_faces_ptr[i * 4 + 2];
                    int v3 = extra_faces_ptr[i * 4 + 3];
                    
                    // 第一个三角形: v0, v1, v2
                    extra_body.faces.push_back(glm::uvec3(v0, v1, v2));
                    // 第二个三角形: v0, v2, v3
                    extra_body.faces.push_back(glm::uvec3(v0, v2, v3));
                }
            } else {
                // 三角形
                for (int i = 0; i < numExtraFaces; i++) {
                    extra_body.faces.push_back(glm::uvec3(
                        extra_faces_ptr[i * 3 + 0],
                        extra_faces_ptr[i * 3 + 1],
                        extra_faces_ptr[i * 3 + 2]
                    ));
                }
            }
            
            // 为额外几何体设置统一的灰色
            glm::vec3 grayColor(0.7f, 0.7f, 0.7f);
            for (size_t i = 0; i < extra_body.faces.size(); i++) {
                extra_body.colors.push_back(grayColor);
            }
            
            data.bodies.push_back(extra_body);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] GetVisualizationData failed: {}", e.what());
    }
    
    return data;
}

void PhysicsBridge::SetExternalForce(const std::string& axis, bool value) {
    try {
        std::string key = "pull_" + axis;
        if (_system_def["external_forces"].contains(key.c_str())) {
            py::module_ jnp = py::module_::import("jax.numpy");
            _system_def["external_forces"][key.c_str()] = jnp.attr("array")(static_cast<double>(value));
        }
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] SetExternalForce failed: {}", e.what());
    }
}

void PhysicsBridge::SetForceStrength(float strength) {
    try {
        if (_system_def["external_forces"].contains("pull_strength")) {
            _system_def["external_forces"]["pull_strength"] = strength;
        }
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] SetForceStrength failed: {}", e.what());
    }
}

} // namespace VCX::Engine::Python::FEM
