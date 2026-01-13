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
    
    // 获取维度信息（pos_dim 与 python 一致）
    _posDim = py::cast<int>(_system.attr("pos_dim"));

    // 加载C++网格数据（无需Python）
    _meshData = Mesh::MeshDataLoader::LoadFEMMesh(problemName);
    _meshData.pos_dim = _posDim;

    // 同步 Python 侧 fixed/unfixed 信息，便于可视化展开 DOF
    try {
        py::dict sys = _system_def.cast<py::dict>();
        if (sys.contains("fixed_inds")) {
            _meshData.fixed_inds = sys["fixed_inds"].cast<std::vector<int>>();
        }
        if (sys.contains("unfixed_inds")) {
            _meshData.unfixed_inds = sys["unfixed_inds"].cast<std::vector<int>>();
        }
        if (sys.contains("fixed_values")) {
            _meshData.fixed_values = sys["fixed_values"].cast<std::vector<double>>();
        }
    } catch (const std::exception& e) {
        spdlog::warn("[PhysicsBridge] sync fixed/unfixed info failed: {}", e.what());
    }

    _fullDim = static_cast<int>(_meshData.Vrest.size() * _posDim);

    spdlog::info("[PhysicsBridge] Loaded C++ mesh: {} verts, {} tris", 
                 _meshData.Vrest.size(), _meshData.E_tri.size());
    
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
        // 子空间模式：初始化为子空间域的初始值
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

void PhysicsBridge::ReloadProblem(const std::string& problemName, const std::string& subspaceModelPath) {
    spdlog::info("[PhysicsBridge] Reloading problem: {}", problemName);
    
    try {
        // 重新加载系统，不需要重新初始化Python
        LoadSystem("fem", problemName, subspaceModelPath);
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] ReloadProblem failed: {}", e.what());
        throw;
    }
}

py::object PhysicsBridge::StateToSystem(const py::object& state) {
    if (_useSubspace) {
        // 应用子空间映射：subspace_apply(system_def, state) -> full_state
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
        // 子空间模式：重置到子空间域的初始值
        double initial_val = py::cast<double>(_subspace_domain_dict["initial_val"]);
        _int_state["q_t"] = jnp.attr("full")(_subspaceDim, initial_val);
        _int_state["q_tm1"] = _int_state["q_t"];
        _int_state["qdot_t"] = jnp.attr("zeros_like")(_int_state["q_t"]);
        spdlog::info("[PhysicsBridge] ResetState (subspace): state.shape={}", _subspaceDim);
    } else {
        // 全空间模式：重置到系统初始位置
        py::object init_pos = _system_def["init_pos"];
        py::object shape = init_pos.attr("shape");
        int size = py::cast<py::tuple>(shape)[0].cast<int>();
        spdlog::info("[PhysicsBridge] ResetState (fullspace): init_pos.shape={}, fullDim={}", size, _fullDim);
        
        _int_state["q_t"] = init_pos;
        _int_state["q_tm1"] = init_pos;
        _int_state["qdot_t"] = jnp.attr("zeros_like")(init_pos);
    }
    
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
        // 获取当前状态（从Python）
        py::object q = StateToSystem(_int_state["q_t"]);
        
        // 转换为C++ vector
        py::array_t<double> q_array = py::cast<py::array_t<double>>(q);
        auto q_buf = q_array.request();
        std::vector<double> q_vec(q_buf.size);
        double* q_ptr = static_cast<double*>(q_buf.ptr);
        for (size_t i = 0; i < q_buf.size; i++) {
            q_vec[i] = q_ptr[i];
        }
        
        // 如果 q 仍是“自由 DOF”长度，先展开到全空间
        const size_t expected_full = _meshData.Vrest.size() * _meshData.pos_dim;
        if (q_vec.size() != expected_full && !_meshData.unfixed_inds.empty() && !_meshData.fixed_inds.empty()) {
            std::vector<double> full_q(expected_full, 0.0);

            const size_t fixed_n = std::min(_meshData.fixed_inds.size(), _meshData.fixed_values.size());
            for (size_t i = 0; i < fixed_n; i++) {
                int idx = _meshData.fixed_inds[i];
                if (idx >= 0 && static_cast<size_t>(idx) < full_q.size()) {
                    full_q[idx] = _meshData.fixed_values[i];
                }
            }

            if (q_vec.size() == _meshData.unfixed_inds.size()) {
                for (size_t i = 0; i < _meshData.unfixed_inds.size(); i++) {
                    int idx = _meshData.unfixed_inds[i];
                    if (idx >= 0 && static_cast<size_t>(idx) < full_q.size()) {
                        full_q[idx] = q_vec[i];
                    }
                }
                q_vec.swap(full_q);
            } else {
                spdlog::warn("[PhysicsBridge] q size {} mismatch with unfixed {}, skip expansion", q_vec.size(), _meshData.unfixed_inds.size());
            }
        }

        // 使用C++生成可视化数据（无需调用Python）
        auto bodies = Mesh::VisualizationBuilder::BuildFEMVisualization(_meshData, q_vec);
        
        // 转换为返回格式
        for (auto& body : bodies) {
            MeshVisualizationData::Body result_body;
            result_body.vertices = body.vertices;
            result_body.faces = body.faces;
            result_body.colors = body.colors;
            data.bodies.push_back(result_body);
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
        
    } catch (const std::exception& e) {
        spdlog::error("[PhysicsBridge] Failed to load subspace model: {}", e.what());
        _useSubspace = false;
        _subspaceDim = _fullDim;
        throw;
    }
}

} // namespace VCX::Engine::Python::FEM
