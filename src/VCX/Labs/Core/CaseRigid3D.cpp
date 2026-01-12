#include "CaseRigid3D.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace VCX::Labs::NeuralPhysicsSubspaces {

CaseRigid3D::CaseRigid3D(const std::string& pythonPath)
    : _pythonPath([&pythonPath]() {
        // 如果提供了路径，直接使用
        if (!pythonPath.empty()) {
            return pythonPath;
        }
        // 否则尝试从CONDA_PREFIX环境变量获取
        const char* conda_prefix = std::getenv("CONDA_PREFIX");
        if (conda_prefix) {
            return std::string(conda_prefix) + "/bin/python";
        }
        // 如果都没有，返回空字符串（使用系统Python）
        return std::string("");
    }()),
      _program(
          Engine::GL::UniqueProgram({
              Engine::GL::SharedShader("assets/shaders/lit_flat_color.vert"),
              Engine::GL::SharedShader("assets/shaders/lit_flat_color.frag") })),
      _meshItem(Engine::GL::VertexLayout()
          .Add<glm::vec3>("position", Engine::GL::DrawFrequency::Stream, 0)
          .Add<glm::vec3>("color", Engine::GL::DrawFrequency::Stream, 1), 
          Engine::GL::PrimitiveType::Triangles) {
    
    // 设置相机
    _cameraManager.AutoRotate = false;
    _cameraManager.Save(_camera);
    
    // 加载默认问题并初始化Python桥接
    LoadProblem(_currentProblem);
    
    spdlog::info("[CaseRigid3D] Initialized with problem: {}", _currentProblem);
}

void CaseRigid3D::LoadProblem(const std::string& problemName) {
    spdlog::info("[CaseRigid3D] Loading problem: {}", problemName);
    
    // 如果已经有物理引擎，重新加载问题；否则创建新的，不然会出现无法重复构建导致画面无法更新的问题
    if (_physics) {
        _physics->ReloadProblem(problemName);
    } else {
        _physics = std::make_unique<Engine::Python::Rigid3D::PhysicsBridge>(
            _pythonPath, "Rigid3d", problemName, ""
        );
    }
    
    // 获取系统信息
    _fullDim = _physics->GetFullDim();
    _subspaceDim = _physics->GetSubspaceDim();
    _useSubspace = _physics->UseSubspace();
    
    if (_useSubspace) {
        _latentState = _physics->GetState();
    }
    
    // 加载外部参数
    LoadExternalParams();
    
    _currentProblem = problemName;
    _needReload = false;
    
    spdlog::info("[CaseRigid3D] Problem loaded:  {} (dim: {})", problemName, _fullDim);
}

void CaseRigid3D::LoadExternalParams() {
    // 初始化外部参数
    auto ext = _physics->GetExternalForce();
    _externalParams.hasExternalForces = ext.hasExternalForces;
    _externalParams.forceStrengthMin = static_cast<float>(ext.force_strength_minmax.first);
    _externalParams.forceStrengthMax = static_cast<float>(ext.force_strength_minmax.second);
    _externalParams.has_X = ext.has_X;
    _externalParams.has_Y = ext.has_Y;
    _externalParams.has_Z = ext.has_Z;
    _externalParams.forceStrengthX = static_cast<float>(ext.force_strength_x);
    _externalParams.forceStrengthY = static_cast<float>(ext.force_strength_y);
    _externalParams.forceStrengthZ = static_cast<float>(ext.force_strength_z);
    _externalParams.unifiedStrength = 0.0f;
}

void CaseRigid3D::ApplyExternalParams() {
    // 将参数应用到 Python
    if (_externalParams.hasExternalForces) {
        // 如果用户修改了统一强度，则同步三个轴
        if (_externalParams.unifiedStrength != 0.0f) {
            _physics->SetForceStrength(_externalParams.unifiedStrength);
        }
        _physics->SetExternalForce("x", _externalParams.forceStrengthX);
        _physics->SetExternalForce("y", _externalParams.forceStrengthY);
        _physics->SetExternalForce("z", _externalParams.forceStrengthZ);
    } else {
        float mid = 0.5f * (_externalParams.forceStrengthMin + _externalParams.forceStrengthMax);
        _physics->SetForceStrength(mid);
        _physics->SetExternalForce("x", 0.0f);
        _physics->SetExternalForce("y", 0.0f);
        _physics->SetExternalForce("z", 0.0f);
    }
}

void CaseRigid3D::OnSetupPropsUI() {
    // ==================== 问题选择器 ====================
    if (ImGui::CollapsingHeader("Problem Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* problemItems[] = {"klann", "stewart"};
        
        if (ImGui::Combo("Problem Type", &_currentProblemIdx, problemItems, IM_ARRAYSIZE(problemItems))) {
            _needReload = true;
        }
        
        if (_needReload) {
            LoadProblem(_problemNames[_currentProblemIdx]);
        }
        
        ImGui::Spacing();
        
        // 问题描述，方便理解
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Description:");
        ImGui::Indent();
        if (_currentProblem == "klann") {
            ImGui::BulletText("A 3D mechanism with a complex kinematic structure");
        } else if (_currentProblem == "stewart") {
            ImGui::BulletText("A Stewart platform with six degrees of freedom");
        }
        ImGui::Unindent();
    }
    ImGui::Spacing();
    
    // ==================== 系统信息 ====================
    if (ImGui::CollapsingHeader("System Information", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("System Type: Rigid 3D");
        ImGui::Text("Problem: %s", _currentProblem.c_str());
        ImGui::Text("Full Dimension: %d", _fullDim);
        
        if (_useSubspace) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.3f, 1.0f), "Subspace Mode Enabled");
            ImGui::Text("Subspace Dimension: %d", _subspaceDim);
            ImGui::Text("Compression Ratio: %.1f%%", 
                       100.0f * _subspaceDim / std::max(_fullDim, 1));
        }
    }
    ImGui::Spacing();
    
    // ==================== 子空间探索 ====================
    if (_useSubspace && ImGui::CollapsingHeader("Latent Space Explorer")) {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.4f, 1.0f), 
                          "Explore the learned low-dimensional manifold");
        ImGui::Spacing();
        
        bool changed = false;
        
        for (size_t i = 0; i < _latentState.size(); ++i) {
            char label[32];
            snprintf(label, sizeof(label), "z[%zu]", i);
            
            float val = static_cast<float>(_latentState[i]);
            if (ImGui::SliderFloat(label, &val, -2.0f, 2.0f)) {
                _latentState[i] = val;
                changed = true;
            }
        }
        
        if (changed) {
            _physics->SetState(_latentState);
        }
        
        ImGui::Spacing();
        if (ImGui::Button("Reset to Origin", ImVec2(-1, 0))) {
            std::fill(_latentState.begin(), _latentState.end(), 0.0);
            _physics->SetState(_latentState);
        }
    }
    ImGui::Spacing();
    
    // ==================== 操控部分 ====================
    if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen)) {
        // 势能显示
        if (_evalEnergyEvery) {
            ImGui::Text("Potential Energy:");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "%.6f", _currentEnergy);
        }
        ImGui::Checkbox("Evaluate Energy", &_evalEnergyEvery);
        
        ImGui::Separator();
        
        // 控制按钮
        if (ImGui::Button("Reset System", ImVec2(120, 0))) {
            _physics->ResetState();
            if (_useSubspace) {
                _latentState = _physics->GetState();
            }
        }
        ImGui::SameLine();
        
        if (ImGui::Button("Stop Velocity", ImVec2(120, 0))) {
            _physics->StopVelocity();
        }
        
        ImGui::Spacing();
        
        // 运行控制
        ImGui::Checkbox("Run Simulation", &_runSimulation);
        
        ImGui::SameLine();
        if (ImGui::Button("Single Step", ImVec2(100, 0))) {
            _physics->Timestep(_timestep);
            if (_useSubspace) {
                _latentState = _physics->GetState();
            }
        }
        
        ImGui::Separator();
        ImGui::SliderFloat("Timestep (dt)", &_timestep, 0.001f, 0.2f, "%.4f");
    }
    ImGui::Spacing();
    
    // ==================== 外力设置 ====================
        if (_externalParams.hasExternalForces && ImGui::CollapsingHeader("External Forces")) {
            ImGui::TextColored(ImVec4(0.9f, 0.5f, 0.5f, 1.0f), "Apply external loads to the system");
            ImGui::Spacing();
        
            bool changed = false;
        
            ImGui::Text("Per-axis strength:");
            if (_externalParams.has_X) {
                changed |= ImGui::SliderFloat("Fx", &_externalParams.forceStrengthX,
                                              _externalParams.forceStrengthMin,
                                              _externalParams.forceStrengthMax, "%.4f");
            }
            if (_externalParams.has_Y) {
                changed |= ImGui::SliderFloat("Fy", &_externalParams.forceStrengthY,
                                              _externalParams.forceStrengthMin,
                                              _externalParams.forceStrengthMax, "%.4f");
            }
            if (_externalParams.has_Z) {
                changed |= ImGui::SliderFloat("Fz", &_externalParams.forceStrengthZ,
                                              _externalParams.forceStrengthMin,
                                              _externalParams.forceStrengthMax, "%.4f");
            }

            ImGui::Separator();
            ImGui::Text("Unified strength (optional):");
            changed |= ImGui::SliderFloat("F all", &_externalParams.unifiedStrength,
                                          _externalParams.forceStrengthMin,
                                          _externalParams.forceStrengthMax, "%.4f");
        
            if (changed) {
                ApplyExternalParams();
            }
        }
    ImGui::Spacing();
    
    // ==================== 渲染设置 ====================
    if (ImGui::CollapsingHeader("Visualization")) {
        ImGui::ColorEdit3("Mesh Color", glm::value_ptr(_meshColor));
        ImGui::Checkbox("Wireframe Mode", &_showWireframe);
        ImGui::SliderFloat("Transparency", &_transparency, 0.0f, 1.0f);
    }
    ImGui::Spacing();
    
    // ==================== 帮助信息 ====================
    if (ImGui::CollapsingHeader("Controls & Help")) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Camera Controls:");
        ImGui::Indent();
        ImGui::BulletText("Left Mouse: Rotate view");
        ImGui::BulletText("Middle Mouse: Pan camera");
        ImGui::BulletText("Mouse Wheel: Zoom in/out");
        ImGui::Unindent();
        
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "TODO:Keyboard Shortcuts:");
        ImGui::Indent();
        ImGui::BulletText("Space: Start/Stop simulation");
        ImGui::BulletText("R: Reset system");
        ImGui::Unindent();
    }
}

Common::CaseRenderResult CaseRigid3D::OnRender(
    std::pair<std::uint32_t, std::uint32_t> const desiredSize) {
    
    // 执行仿真
    if (_runSimulation) {
        _physics->Timestep(_timestep);
        if (_useSubspace) {
            _latentState = _physics->GetState();
        }
    }
    
    // 计算能量
    if (_evalEnergyEvery) {
        _currentEnergy = _physics->GetPotentialEnergy();
    }
    
    // 更新可视化
    UpdateVisualization();
    
    // 渲染
    RenderScene(desiredSize);
    
    return Common:: CaseRenderResult {
        .Fixed = false,
        .Flipped = true,
        .Image = _frame.GetColorAttachment(),
        .ImageSize = desiredSize,
    };
}

void CaseRigid3D::UpdateVisualization() {
    auto meshData = _physics->GetVisualizationData();
    
    if (! meshData.bodies.empty()) {
        std::vector<glm::vec3> allFaceVertices;
        std::vector<glm::vec3> allFaceColors;
        std::vector<uint32_t> allIndices;
        
        // 合并所有bodies
        for (const auto& body : meshData.bodies) {
            uint32_t vertexOffset = allFaceVertices.size();
            
            // 如果有face colors，为每个面创建独立的顶点
            if (!body.colors.empty()) {
                for (size_t i = 0; i < body.faces.size(); i++) {
                    const auto& face = body.faces[i];
                    const glm::vec3& faceColor = body.colors[i];
                    
                    // 为这个面创建三个独立的顶点
                    uint32_t baseIdx = allFaceVertices.size();
                    allFaceVertices.push_back(body.vertices[face.x]);
                    allFaceVertices.push_back(body.vertices[face.y]);
                    allFaceVertices.push_back(body.vertices[face.z]);
                    
                    // 三个顶点使用相同的面颜色
                    allFaceColors.push_back(faceColor);
                    allFaceColors.push_back(faceColor);
                    allFaceColors.push_back(faceColor);
                    
                    // 添加索引
                    allIndices.push_back(baseIdx + 0);
                    allIndices.push_back(baseIdx + 1);
                    allIndices.push_back(baseIdx + 2);
                }
            } else {
                // 没有颜色信息，使用原始顶点和默认颜色
                for (const auto& vert : body.vertices) {
                    allFaceVertices.push_back(vert);
                    allFaceColors.push_back(glm::vec3(0.8f, 0.6f, 0.4f));
                }
                
                for (const auto& face : body.faces) {
                    allIndices.push_back(vertexOffset + face.x);
                    allIndices.push_back(vertexOffset + face.y);
                    allIndices.push_back(vertexOffset + face.z);
                }
            }
        }
        
        _meshItem.UpdateVertexBuffer("position",
            Engine::make_span_bytes<glm::vec3>(allFaceVertices)
        );
        _meshItem.UpdateVertexBuffer("color",
            Engine::make_span_bytes<glm::vec3>(allFaceColors)
        );
        _meshItem.UpdateElementBuffer(allIndices);
    }
}

void CaseRigid3D::RenderScene(std::pair<std::uint32_t, std::uint32_t> const size) {
    _frame.Resize(size);
    _cameraManager.Update(_camera);
    
    glm::mat4 projection = _camera.GetProjectionMatrix(float(size.first) / size.second);
    glm::mat4 view = _camera.GetViewMatrix();
    
    _program.GetUniforms().SetByName("u_Projection", projection);
    _program.GetUniforms().SetByName("u_View", view);
    _program.GetUniforms().SetByName("u_Model", glm::mat4(1.0f));
    _program.GetUniforms().SetByName("u_Color", glm::vec3(0.8f, 0.6f, 0.4f));
    
    gl_using(_frame);
    glClearColor(0.2f, 0.2f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    
    _meshItem.Draw({ _program. Use() });
}

void CaseRigid3D::OnProcessInput(ImVec2 const & pos) {
    _cameraManager.ProcessInput(_camera, pos);
}

} // namespace VCX:: Labs::NeuralPhysics