#include "Engine/Python/Rigid3DPhysicsBridge.h"
#include "Engine/Python/PythonInterpreter.h"
#include <spdlog/spdlog.h>
#include <iostream>
#include <memory>
#include <cmath>

namespace py = pybind11;
using namespace pybind11::literals;

namespace VCX::Engine::Python::Rigid3D {

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

// ================ 加载物理系统（相关函数可以见rigid3d_modal.py及main_run_system.py） ====================
void PhysicsBridge::LoadSystem(const std::string& systemName,
                               const std::string& problemName,
                               const std::string& subspaceModelPath) {
    
    // 导入必要的Python模块
    py::module_ rigid3d_model = py::module_::import("rigid3d_model");
    py::module_ integrators = py::module_::import("integrators");
    py::module_ np = py::module_::import("numpy");
    py::module_ jnp = py::module_::import("jax.numpy");
    
    // 构建Rigid3D系统
    py::object Rigid3DSystem = rigid3d_model.attr("Rigid3DSystem");
    py::tuple result = Rigid3DSystem.attr("construct")(problemName);
    _system = result[0];
    _system_def = result[1];
    
    // 获取维度信息
    _fullDim = py::cast<int>(_system_def["init_pos"].attr("size"));
    _posDim = py::cast<int>(_system.attr("pos_dim"));
    
    // 检查是否使用子空间
    _useSubspace = !subspaceModelPath.empty();
    
    if (_useSubspace) {
        // 加载子空间模型
        SetSubspaceModel(subspaceModelPath);
    } else {
        _subspaceDim = _fullDim;
    }

    // 初始化积分器
    _int_opts = py::dict();
    _int_state = py::dict();
    
    // 使用隐式近端积分器
    integrators.attr("initialize_integrator")(_int_opts, _int_state, "implicit-proximal");
    
    // 设置初始状态
    if (_useSubspace) {
        // 子空间模式：初始化为子空间域的初始值，注意维度切换
        double initial_val = py::cast<double>(_subspace_domain_dict["initial_val"]);
        _int_state["q_t"] = jnp.attr("full")(_subspaceDim, initial_val);
        _int_state["q_tm1"] = _int_state["q_t"];
        _int_state["qdot_t"] = jnp.attr("zeros_like")(_int_state["q_t"]);
    } else {
        // 全空间模式：使用系统初始位置
        _int_state["q_t"] = _system_def["init_pos"];
        _int_state["q_tm1"] = _system_def["init_pos"];
        _int_state["qdot_t"] = jnp.attr("zeros_like")(_system_def["init_pos"]);
    }
    py::dict forces = _system_def["external_forces"].cast<py::dict>();
    if (forces.contains("force_verts_mask")) _externalForce.force_verts_mask = forces["force_verts_mask"].cast<std::vector<bool>>();

    // force strength min/max
    if (forces.contains("force_strength_minmax")) {
        _externalForce.force_strength_minmax = forces["force_strength_minmax"].cast<std::pair<double,double>>();
    } else {
        _externalForce.force_strength_minmax = std::make_pair(-10.0, 10.0);
    }

    auto read_axis = [&](const char* key, double& outVal, bool& hasFlag) {
        if (forces.contains(key)) {
            hasFlag = true;
            outVal = forces[key].cast<double>();
        } else {
            hasFlag = false;
            outVal = 0.0;
        }
    };
    read_axis("force_strength_x", _externalForce.force_strength_x, _externalForce.has_X);
    read_axis("force_strength_y", _externalForce.force_strength_y, _externalForce.has_Y);
    read_axis("force_strength_z", _externalForce.force_strength_z, _externalForce.has_Z);
    _externalForce.hasExternalForces = _externalForce.has_X || _externalForce.has_Y || _externalForce.has_Z;
    spdlog::info("[PhysicsBridge] System loaded: dim={}, pos_dim={}", _fullDim, _posDim);
}

void PhysicsBridge::ReloadProblem(const std::string& problemName, const std::string& subspaceModelPath) {
    spdlog::info("[PhysicsBridge] Reloading problem: {}", problemName);
    
    try {
        // 重新加载系统，不需要重新初始化Python
        LoadSystem("Rigid3d", problemName, subspaceModelPath);
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] ReloadProblem failed: {}", e.what());
        throw;
    }
}

py::object PhysicsBridge::StateToSystem(const py::object& state) {
    if (_useSubspace) {
        // 应用子空间映射回原系统：subspace_apply(system_def, state) -> full_state
        try {
            return _subspace_apply(_system_def, state);
        } catch (const std::exception& e) {
            spdlog::error("[PhysicsBridge] StateToSystem failed: {}", e.what());
            return state;
        }
    } else {
        return state;
    }
}

void PhysicsBridge::ResetState() {
    py::module_ jnp = py::module_::import("jax.numpy");
    if (_useSubspace) {
        double initial_val = py::cast<double>(_subspace_domain_dict["initial_val"]);
        _int_state["q_t"] = jnp.attr("full")(_subspaceDim, initial_val);
        _int_state["q_tm1"] = _int_state["q_t"];
        _int_state["qdot_t"] = jnp.attr("zeros_like")(_int_state["q_t"]);
    } else {
        _int_state["q_t"] = _system_def["init_pos"];
        _int_state["q_tm1"] = _system_def["init_pos"];
        _int_state["qdot_t"] = jnp.attr("zeros_like")(_system_def["init_pos"]);
    }
    py::dict forces = _system_def["external_forces"].cast<py::dict>();
    if (forces.contains("force_verts_mask")) _externalForce.force_verts_mask = forces["force_verts_mask"].cast<std::vector<bool>>();
    _externalForce.force_strength_minmax = forces.contains("force_strength_minmax")
        ? forces["force_strength_minmax"].cast<std::pair<double,double>>()
        : std::make_pair(-10.0, 10.0);

    auto read_axis_reset = [&](const char* key, double& outVal, bool& hasFlag) {
        if (forces.contains(key)) {
            hasFlag = true;
            outVal = forces[key].cast<double>();
        } else {
            hasFlag = false;
            outVal = 0.0;
        }
    };
    read_axis_reset("force_strength_x", _externalForce.force_strength_x, _externalForce.has_X);
    read_axis_reset("force_strength_y", _externalForce.force_strength_y, _externalForce.has_Y);
    read_axis_reset("force_strength_z", _externalForce.force_strength_z, _externalForce.has_Z);
    _externalForce.hasExternalForces = _externalForce.has_X || _externalForce.has_Y || _externalForce.has_Z;
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
        
        if (_useSubspace) {
            // 子空间模式：传递子空间映射函数和域字典
            _int_state = integrators.attr("timestep")(
                _system,
                _system_def,
                _int_state,
                _int_opts,
                "subspace_fn"_a = _subspace_apply,
                "subspace_domain_dict"_a = _subspace_domain_dict
            );
        } else {
            // 全空间模式：不使用子空间
            _int_state = integrators.attr("timestep")(
                _system,
                _system_def,
                _int_state,
                _int_opts,
                "subspace_fn"_a = py::none(),
                "subspace_domain_dict"_a = py::none()
            );
        }
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
        
        py::object bodies_data = _system.attr("get_visualization_data")(_system, _system_def, q);
        py::list bodies_list = py::cast<py::list>(bodies_data);
        
        // 遍历每个刚体
        for (size_t bid = 0; bid < bodies_list.size(); bid++) {
            py::dict body_dict = py::cast<py::dict>(bodies_list[bid]);
            
            MeshVisualizationData::Body body;
            
            // 获取顶点数据
            py::array_t<double> verts_array = py::cast<py::array_t<double>>(
                np.attr("array")(body_dict["vertices"])
            );
            auto verts_buf = verts_array.request();
            int numVerts = verts_buf.shape[0];
            double* verts_ptr = static_cast<double*>(verts_buf.ptr);
            
            // Rigid3D总是3D的
            for (int i = 0; i < numVerts; i++) {
                body.vertices.push_back(glm::vec3(
                    static_cast<float>(verts_ptr[i * 3 + 0]),
                    static_cast<float>(verts_ptr[i * 3 + 1]),
                    static_cast<float>(verts_ptr[i * 3 + 2])
                ));
            }
            
            // 获取面数据
            py::array_t<int> faces_array = py::cast<py::array_t<int>>(
                np.attr("array")(body_dict["faces"])
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
            
            // 为每个刚体自动分配不同颜色（模拟 Polyscope 行为）
            glm::vec3 bodyColor;
            float hue = static_cast<float>(bid) / static_cast<float>(bodies_list.size()) * 360.0f;
            // 使用 HSV 转 RGB 来生成均匀分布的颜色
            float h = hue / 60.0f;
            float c = 0.7f;  // 饱和度
            float x = c * (1.0f - std::abs(std::fmod(h, 2.0f) - 1.0f));
            float m = 0.3f;  // 亮度偏移
            
            if (h < 1.0f)      bodyColor = glm::vec3(c, x, 0) + m;
            else if (h < 2.0f) bodyColor = glm::vec3(x, c, 0) + m;
            else if (h < 3.0f) bodyColor = glm::vec3(0, c, x) + m;
            else if (h < 4.0f) bodyColor = glm::vec3(0, x, c) + m;
            else if (h < 5.0f) bodyColor = glm::vec3(x, 0, c) + m;
            else               bodyColor = glm::vec3(c, 0, x) + m;
            
            // 为这个刚体的所有面设置相同的颜色
            for (int i = 0; i < numFaces; i++) {
                body.colors.push_back(bodyColor);
            }
            
            data.bodies.push_back(body);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] GetVisualizationData failed: {}", e.what());
    }
    
    return data;
}

void PhysicsBridge::SetExternalForce(const std::string& axis, float strength) {
    try {
        std::string key = "force_strength_" + axis;
        if (_system_def["external_forces"].contains(key.c_str())) {
            _system_def["external_forces"][key.c_str()] = strength;
        }
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] SetExternalForce failed: {}", e.what());
    }
}

void PhysicsBridge::SetForceStrength(float strength) {
    try {
        if (_system_def["external_forces"].contains("force_strength_minmax")) {
            // 更新统一的强度中心值：将x/y/z都设置为该值
            _system_def["external_forces"]["force_strength_x"] = strength;
            _system_def["external_forces"]["force_strength_y"] = strength;
            _system_def["external_forces"]["force_strength_z"] = strength;
        }
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] SetForceStrength failed: {}", e.what());
    }
}

void PhysicsBridge::SetSubspaceModel(const std::string& modelPath) {
    spdlog::info("[PhysicsBridge] Loading subspace model: {}", modelPath);
    
    try {
        py::module_ subspace = py::module_::import("subspace");
        py::module_ layers = py::module_::import("layers");
        py::module_ eqx = py::module_::import("equinox");
        py::module_ jax = py::module_::import("jax");
        py::module_ np = py::module_::import("numpy");
        py::module_ jnp = py::module_::import("jax.numpy");
        py::module_ json_module = py::module_::import("json");
        
        // 加载模型配置
        py::object json_load = json_module.attr("load");
        py::object open_fn = py::module_::import("builtins").attr("open");
        py::object json_file = open_fn(modelPath + ".json", "r");
        py::object spec = json_load(json_file);
        json_file.attr("close")();
        
        // 创建模型
        py::object model = layers.attr("create_model")(spec);
        py::tuple partition_result = eqx.attr("partition")(model, eqx.attr("is_array"));
        py::object model_static = partition_result[1];
        
        // 加载模型参数
        py::object model_params = eqx.attr("tree_deserialise_leaves")(modelPath + ".eqx", model);
        
        // 加载元数据
        py::object info = np.attr("load")(modelPath + "_info.npy", "allow_pickle"_a=true).attr("item")();
        _subspaceDim = py::cast<int>(info["subspace_dim"]);
        
        // 获取子空间域信息
        py::object domain_type = info["subspace_domain_type"];
        _subspace_domain_dict = subspace.attr("get_subspace_domain_dict")(domain_type);
        _t_schedule_final = py::cast<double>(info["t_schedule_final"]);
        
        // 创建apply函数，也就是子空间映射到全空间的函数
        py::object jit_fn = jax.attr("jit");
        auto apply_fn = [model_params, model_static, eqx, jnp, this](py::object system_def, py::object x) {
            // 从system_def中获取条件参数
            py::object cond_params = system_def["cond_param"];
            
            // 组合模型
            py::object m = eqx.attr("combine")(model_params, model_static);
            
            // 连接输入：[x, cond_params]，x为子空间物体状态
            py::object concatenated = jnp.attr("concatenate")(py::make_tuple(x, cond_params), "axis"_a=-1);
            
            // 运行模型
            return m(concatenated, "t_schedule"_a=_t_schedule_final);
        };
        
        // 使用lambda包装并JIT编译
        _subspace_apply = py::cpp_function(apply_fn);
        _subspace_apply = jit_fn(_subspace_apply);
        
        _useSubspace = true;
        
        spdlog::info("[PhysicsBridge] Subspace model loaded successfully: {}D -> {}D", _subspaceDim, _fullDim);
        
        // 重置状态到子空间
        ResetState();
        
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] Failed to load subspace model: {}", e.what());
        _useSubspace = false;
        _subspaceDim = _fullDim;
        throw;
    }
}

} // namespace VCX::Engine::Python::Rigid3D
