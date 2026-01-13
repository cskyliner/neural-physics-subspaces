#include "CaseRigid3DSubspaces.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace VCX::Labs::NeuralPhysicsSubspaces {

CaseRigid3DSpaces::CaseRigid3DSpaces(const std::string& pythonPath)
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
    
    // 初始化物理引擎，先展示默认问题，避免全空，但是因为模型未知，所以后续会重新加载
    _physics = std::make_unique<Engine::Python::Rigid3D::PhysicsBridge>(
        _pythonPath, "rigid3d", _currentProblem, ""
    );
    
    // 获取系统信息
    _fullDim = _physics->GetFullDim();
    _subspaceDim = _physics->GetSubspaceDim();
    _useSubspace = _physics->UseSubspace();
    // 重置物理状态
    _physics->ResetState();
    // 加载外部参数
    LoadExternalParams();
    // 设置相机
    _cameraManager.AutoRotate = false;
    _cameraManager.Save(_camera);
    // 初始化checkpoint范围
    auto& loader = ModelLoader::GetInstance();
    _currentRange = loader.GetCheckpointRange(ModelType::Rigid3D, _currentProblem, _currentDim, _currentWexp);
    if (_currentRange.hasFinal) {
        _selectedCheckpoint = -1;
    } else if (_currentRange.minIndex >= 0) {
        _selectedCheckpoint = _currentRange.minIndex;
    }
    
    // 初始化可视化
    try {
        UpdateVisualization();
        spdlog::info("[CaseRigid3DSpaces] Initialized with problem: {}", _currentProblem);
    } catch (const std::exception& e) {
        spdlog::error("[CaseRigid3DSpaces] Initial visualization failed: {}", e.what());
    }
}


void CaseRigid3DSpaces::LoadExternalParams() {
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

void CaseRigid3DSpaces::ApplyExternalParams() {
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

void CaseRigid3DSpaces::OnSetupPropsUI() {
    // ==================== 问题选择器 ====================
    if (ImGui::CollapsingHeader("Problem Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* problemItems[] = {"klann", "stewart"};
        
        if (ImGui::Combo("Problem Type", &_currentProblemIdx, problemItems, IM_ARRAYSIZE(problemItems))) {
            _currentProblem = _problemNames[_currentProblemIdx];
            
            // 重新加载基础问题（无子空间）
            spdlog::info("[CaseRigid3DSpaces] Problem changed to: {}, reloading base system", _currentProblem);
            try {
                _physics->ReloadProblem(_currentProblem, "");  // Empty path = no subspace
                _useSubspace = false;
                _fullDim = _physics->GetFullDim();
                UpdateVisualization();
            } catch (const std::exception& e) {
                spdlog::error("[CaseRigid3DSpaces] Failed to reload base problem: {}", e.what());
            }
        }
        
        ImGui::Spacing();
        
        // 问题描述，方便理解
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Description:");
        ImGui::Indent();
        if (_currentProblem == "klann") {
            ImGui::BulletText("A 3D mechanism");
            ImGui::BulletText("Complex kinematic structure");
        } else if (_currentProblem == "stewart") {
            ImGui::BulletText("A Stewart platform");
            ImGui::BulletText("Six degrees of freedom");
        }
        ImGui::Unindent();
    }
    ImGui::Spacing();
    // ==================== 模型加载器 ====================
    if (ImGui::CollapsingHeader("Model Loader", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "Load Neural Subspace Models");
        
        // 获取可用维度
        auto& loader = ModelLoader::GetInstance();
        std::vector<int> availableDims = loader.GetDimsForTypeAndProblem(ModelType::Rigid3D, _currentProblem);
        
        // 维度选择
        if (!availableDims.empty()) {
            ImGui::Separator();
            ImGui::Text("Subspace Dimension:");
            ImGui::Indent();
            
            for (int dim : availableDims) {
                char label[32];
                snprintf(label, sizeof(label), "Dim %d", dim);
                if (ImGui::RadioButton(label, _currentDim == dim)) {
                    _currentDim = dim;
                    // 获取可用的wexp值，选择第一个作为默认值
                    std::vector<std::string> availableWexps = loader.GetWexpsForTypeAndProblemDim(ModelType::Rigid3D, _currentProblem, _currentDim);
                    if (!availableWexps.empty()) {
                        _currentWexp = availableWexps[0];
                    }
                    // 更新checkpoint范围
                    _currentRange = loader.GetCheckpointRange(ModelType::Rigid3D, _currentProblem, _currentDim, _currentWexp);
                    // 默认选择final（如果有）或最小索引
                    if (_currentRange.hasFinal) {
                        _selectedCheckpoint = -1;
                    } else if (_currentRange.minIndex >= 0) {
                        _selectedCheckpoint = _currentRange.minIndex;
                    }
                }
                ImGui::SameLine();
            }
            ImGui::NewLine();
            ImGui::Unindent();
        }
        
        // Wexp选择
        std::vector<std::string> availableWexps = loader.GetWexpsForTypeAndProblemDim(ModelType::Rigid3D, _currentProblem, _currentDim);
        if (!availableWexps.empty() && availableWexps.size() >= 1) {
            ImGui::Separator();
            ImGui::Text("Hyperparameter (wexp):");
            ImGui::Indent();
            
            for (const auto& wexp : availableWexps) {
                std::string displayLabel = wexp;  // e.g., "wexp1" or "wexp0.5"
                if (ImGui::RadioButton(displayLabel.c_str(), _currentWexp == wexp)) {
                    _currentWexp = wexp;
                    // 更新checkpoint范围
                    _currentRange = loader.GetCheckpointRange(ModelType::Rigid3D, _currentProblem, _currentDim, _currentWexp);
                    // 默认选择final（如果有）或最小索引
                    if (_currentRange.hasFinal) {
                        _selectedCheckpoint = -1;
                    } else if (_currentRange.minIndex >= 0) {
                        _selectedCheckpoint = _currentRange.minIndex;
                    }
                }
                ImGui::SameLine();
            }
            ImGui::NewLine();
            ImGui::Unindent();
        }
        
        // Checkpoint选择
        if (_currentRange.minIndex >= 0 || _currentRange.hasFinal) {
            ImGui::Separator();
            ImGui::Text("Model Checkpoint:");
            ImGui::Indent();
            
            // Final模型
            if (_currentRange.hasFinal) {
                if (ImGui::RadioButton("Final Model", _selectedCheckpoint == -1)) {
                    _selectedCheckpoint = -1;
                }
            }
            
            // 使用滑块选择checkpoint
            if (_currentRange.minIndex >= 0 && _currentRange.maxIndex >= 0) {
                ImGui::Text("Checkpoint Range: %04d - %04d", _currentRange.minIndex, _currentRange.maxIndex);
                
                int currentCheckpoint = (_selectedCheckpoint >= 0) ? _selectedCheckpoint : _currentRange.minIndex;
                if (ImGui::SliderInt("Checkpoint Index", &currentCheckpoint, 
                                    _currentRange.minIndex, _currentRange.maxIndex,
                                    "save%04d")) {
                    _selectedCheckpoint = currentCheckpoint;
                }
                
                // 显示当前选中checkpoint的实际sigma值
                auto& loader = ModelLoader::GetInstance();
                std::string sigma = loader.GetSigmaForCheckpoint(ModelType::Rigid3D, _currentProblem, _currentDim, _currentWexp, currentCheckpoint);
                ImGui::Text("Sigma: %s", sigma.c_str());
            }
            
            ImGui::Unindent();
        }
        
        // 加载按钮
        ImGui::Separator();
        if (ImGui::Button("Load Selected Model", ImVec2(-1, 0))) {
            LoadSelectedModel();
        }
        
        if (_useSubspace) {
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "Subspace model loaded");
            ImGui::Text("Latent Dim: %d", _subspaceDim);
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
            _physics->ResetState();
            if (_useSubspace) {
                _latentState = _physics->GetState();
            }
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
        if (ImGui::Button("Reset System", ImVec2(200, 0))) {
            _physics->ResetState();
            if (_useSubspace) {
                _latentState = _physics->GetState();
            }
        }
        ImGui::SameLine();
        
        if (ImGui::Button("Stop Velocity", ImVec2(200, 0))) {
            _physics->StopVelocity();
        }
        
        ImGui::Spacing();
        
        // 运行控制
        ImGui::Checkbox("Run Simulation", &_runSimulation);
        
        ImGui::SameLine();
        if (ImGui::Button("Single Step", ImVec2(200, 0))) {
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
            changed |= ImGui::SliderFloat("For all", &_externalParams.unifiedStrength,
                                          _externalParams.forceStrengthMin,
                                          _externalParams.forceStrengthMax, "%.4f");
        
            if (changed) {
                ApplyExternalParams();
            }
        }
    ImGui::Spacing();
    
    // ==================== 渲染设置 ====================
    if (ImGui::CollapsingHeader("Visualization")) {
        ImGui::Checkbox("Wireframe Mode", &_showWireframe);
        ImGui::SliderFloat("Transparency", &_transparency, 0.0f, 1.0f);
    }
    ImGui::Spacing();
    
    // ==================== 帮助信息 ====================
    if (ImGui::CollapsingHeader("Information & Control Help")) {
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
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Camera Controls:");
        ImGui::Indent();
        ImGui::BulletText("Left Mouse: Rotate view");
        // ImGui::BulletText("Middle Mouse: Pan camera");
        ImGui::BulletText("Mouse Wheel: Zoom in/out");
        ImGui::Unindent();
    }
}

Common::CaseRenderResult CaseRigid3DSpaces::OnRender(
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

void CaseRigid3DSpaces::LoadSelectedModel() {
    auto& loader = ModelLoader::GetInstance();
    
    // 构建模型路径
    std::string modelPath = loader.BuildModelPath(ModelType::Rigid3D, _currentProblem, _currentDim, _currentWexp, _selectedCheckpoint);
    
    spdlog::info("[CaseRigid3DSpaces] Attempting to load model: {}", modelPath);
    spdlog::info("[CaseRigid3DSpaces] Problem: {}, Dim: {}, Wexp: {}, Checkpoint: {}", _currentProblem, _currentDim, _currentWexp, _selectedCheckpoint);
    
    // 检查_physics是否存在
    if (!_physics) {
        spdlog::error("[CaseRigid3DSpaces] Physics engine not initialized!");
        return;
    }
    
    // 加载子空间模型
    try {
        // 先重新加载Rigid3D问题，确保system_def匹配子空间模型
        spdlog::info("[CaseRigid3DSpaces] Reloading Rigid3D problem to match subspace model");
        _physics->ReloadProblem(_currentProblem, modelPath);
        
        // 更新系统信息
        _fullDim = _physics->GetFullDim();
        _subspaceDim = _physics->GetSubspaceDim();
        _useSubspace = _physics->UseSubspace();
        
        if (_useSubspace) {
            _latentState.resize(_subspaceDim, 0.0);
            // 重置状态以确保正确初始化
            _physics->ResetState();
            _latentState = _physics->GetState();
        }
        
        spdlog::info("[CaseRigid3DSpaces] Model loaded successfully");
        spdlog::info("[CaseRigid3DSpaces] Subspace dimension: {}, Full dimension: {}", _subspaceDim, _fullDim);
        
        // 重新加载外部参数
        LoadExternalParams();
        
        // 更新可视化
        UpdateVisualization();
    } catch (const std::exception& e) {
        spdlog::error("[CaseRigid3DSpaces] Failed to load model: {}", e.what());
        _useSubspace = false;
    }
}

void CaseRigid3DSpaces::UpdateVisualization() {
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

void CaseRigid3DSpaces::RenderScene(std::pair<std::uint32_t, std::uint32_t> const size) {
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
    // 根据透明度启用 alpha 混合
    if (_transparency < 1.0f) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    } else {
        glDisable(GL_BLEND);
    }
    
    // 根据 _showWireframe 设置多边形模式
    if (_showWireframe) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    _meshItem.Draw({ _program. Use() });
}

void CaseRigid3DSpaces::OnProcessInput(ImVec2 const & pos) {
    _cameraManager.ProcessInput(_camera, pos);
}

} // namespace VCX:: Labs::NeuralPhysicsSubspaces