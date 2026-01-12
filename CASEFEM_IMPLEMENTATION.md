# CaseFEM Python桥接和OpenGL渲染实现指南

## 1. Python桥接 (PhysicsBridge)

### 已实现的功能

**PhysicsBridge.h/cpp** 实现了C++到Python的桥接：

```cpp
// 初始化Python环境并加载FEM系统
PhysicsBridge(pythonPath, "fem", "bistable", "");

// 仿真控制
void Timestep(float dt);           // 执行一个时间步
void ResetState();                  // 重置状态
void StopVelocity();               // 停止速度

// 数据获取
MeshVisualizationData GetVisualizationData();  // 获取可视化网格数据
double GetPotentialEnergy();                    // 获取势能
```

### Python调用流程

1. **加载Python模块**: 
   ```cpp
   py::module_ fem_model = py::module_::import("fem_model");
   ```

2. **构建FEM系统**:
   ```cpp
   py::tuple result = FEMSystem.attr("construct")(problemName);
   _system = result[0];      // FEM系统对象
   _system_def = result[1];  // 系统定义字典
   ```

3. **时间积分**:
   ```cpp
   integrators.attr("timestep")(system, system_def, int_state, int_opts);
   ```

4. **获取可视化数据**:
   ```cpp
   // 获取当前位置
   py::object pos = _system.attr("get_full_position")(system, system_def, q);
   // 获取网格面
   py::object faces = mesh["boundary_triangles"];
   ```

## 2. OpenGL渲染使用flat shader

### Shader文件
- **flat.vert**: 顶点着色器，传递位置并应用MVP矩阵
- **flat.frag**: 片段着色器，使用统一颜色渲染

### 渲染流程

```cpp
// 1. 初始化程序
_program = Engine::GL::UniqueProgram({
    Engine::GL::SharedShader("assets/shaders/flat.vert"),
    Engine::GL::SharedShader("assets/shaders/flat.frag")
});

// 2. 创建渲染项 (使用三角形列表)
_meshItem = Engine::GL::UniqueIndexedRenderItem(
    Engine::GL::VertexLayout()
        .Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0),
    Engine::GL::PrimitiveType::Triangles
);

// 3. 更新顶点数据
auto meshData = _physics->GetVisualizationData();
const auto& body = meshData.bodies[0];

// 上传顶点
_meshItem.UpdateVertexBuffer("position",
    Engine::make_span_bytes<glm::vec3>(body.vertices));

// 上传索引
std::vector<uint32_t> indices;
for (const auto& face : body.faces) {
    indices.push_back(face.x);
    indices.push_back(face.y);
    indices.push_back(face.z);
}
_meshItem.UpdateElementBuffer(indices);

// 4. 设置uniform并渲染
_program.GetUniforms().SetByName("u_Projection", projection);
_program.GetUniforms().SetByName("u_View", view);
_program.GetUniforms().SetByName("u_Color", meshColor);

glEnable(GL_DEPTH_TEST);
_meshItem.Draw({ _program.Use() });
```

## 3. 数据流程

```
Python FEM模拟
    ↓
fem_model.py (计算位置)
    ↓
PhysicsBridge::GetVisualizationData()
    ↓  (转换numpy→glm::vec3)
MeshVisualizationData
    ↓
CaseFEM::UpdateVisualization()
    ↓  (上传到GPU)
OpenGL Buffer
    ↓
flat shader渲染
```

## 4. 编译配置

### xmake.lua需要的修改

```lua
-- 在engine target添加
add_packages("pybind11", { public = true })

-- 可能需要链接Python
if is_plat("linux") then
    add_syslinks("python3.10")  -- 根据你的Python版本
end
```

### 编译命令

```bash
xmake config --yes
xmake build
```

## 5. 使用示例

```cpp
// 创建CaseFEM实例 (在App.h中)
CaseFEM _caseFEM{"./python"};

// 在构造函数中
CaseFEM::CaseFEM(const std::string& pythonPath) {
    // 初始化OpenGL
    _program = Engine::GL::UniqueProgram({
        Engine::GL::SharedShader("assets/shaders/flat.vert"),
        Engine::GL::SharedShader("assets/shaders/flat.frag")
    });
    
    // 初始化Python桥接
    _physics = std::make_unique<Engine::Python::PhysicsBridge>(
        pythonPath, "fem", "bistable", ""
    );
}

// 在OnRender中
void CaseFEM::OnRender() {
    // 执行物理模拟
    if (_runSimulation) {
        _physics->Timestep(0.05f);
    }
    
    // 更新可视化
    auto meshData = _physics->GetVisualizationData();
    _meshItem.UpdateVertexBuffer("position", ...);
    _meshItem.UpdateElementBuffer(...);
    
    // 渲染
    _program.GetUniforms().SetByName("u_Color", _meshColor);
    _meshItem.Draw({ _program.Use() });
}
```

## 6. 潜在问题和解决方案

### Python路径问题
```cpp
// 在PhysicsBridge构造函数中添加路径
py::module_ sys = py::module_::import("sys");
sys.attr("path").attr("append")("./python");
```

### Numpy数组转换
```cpp
// numpy → C++
py::array_t<double> arr = py::cast<py::array_t<double>>(numpy_obj);
auto buf = arr.request();
double* ptr = static_cast<double*>(buf.ptr);

// 转为glm::vec3
for (int i = 0; i < numVerts; i++) {
    vertices.push_back(glm::vec3(
        ptr[i*3+0], ptr[i*3+1], ptr[i*3+2]
    ));
}
```

### 2D vs 3D处理
```cpp
if (dim == 2) {
    // 2D: 添加z=0
    body.vertices.push_back(glm::vec3(x, y, 0.0f));
    // 使用mesh["E"]作为三角形
} else {
    // 3D: 直接使用
    body.vertices.push_back(glm::vec3(x, y, z));
    // 使用mesh["boundary_triangles"]
}
```

## 7. 调试建议

1. **Python错误**: 捕获并打印Python异常
```cpp
try {
    // Python调用
} catch (const py::error_already_set& e) {
    spdlog::error("Python error: {}", e.what());
}
```

2. **可视化检查**: 确认数据格式
```cpp
spdlog::info("Vertices: {}, Faces: {}", 
             body.vertices.size(), body.faces.size());
```

3. **Shader调试**: 检查uniform传递
```cpp
spdlog::info("Projection: {}", glm::to_string(projection));
```
