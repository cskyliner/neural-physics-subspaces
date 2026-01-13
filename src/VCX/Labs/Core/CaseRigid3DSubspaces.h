#pragma once

#include "Engine/app.h"
#include "Engine/Python/Rigid3DPhysicsBridge.h"
#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/OrbitCameraManager.h"
#include "ModelLoader.h"
#include <memory>
#include <vector>
#include <string>

namespace VCX::Labs::NeuralPhysicsSubspaces {

class CaseRigid3DSpaces : public Common::ICase {
private:
    // Python 物理引擎
    std::unique_ptr<Engine::Python::Rigid3D::PhysicsBridge> _physics;
    std::string _pythonPath;
    
    // 问题与模型选择
    std::vector<std::string> _problemNames = {"klann", "stewart"};
    int _currentProblemIdx = 0;
    std::string _currentProblem = "klann";
    int _currentDim = 8;
    std::string _currentWexp = "wexp1";
    int _selectedCheckpoint = -1;  // -1 for final
    CheckpointRange _currentRange;
    bool _needReload = false;
    
    
    // 系统信息
    int _fullDim = 0;
    int _subspaceDim = 0;
    bool _useSubspace = false;
    
    // 仿真控制
    bool _runSimulation = false;
    bool _evalEnergyEvery = true;
    float _timestep = 0.05f;
    double _currentEnergy = 0.0;
    
    // 子空间
    std:: vector<double> _latentState;
    
    // 外部参数（从 Python system_def 获取）
    struct ExternalParams {
        bool hasExternalForces = false;
        float forceStrengthMin = -10.0f;
        float forceStrengthMax = 10.0f;
        float forceStrengthX = 0.0f;
        float forceStrengthY = 0.0f;
        float forceStrengthZ = 0.0f;
        bool has_X = false;
        bool has_Y = false;
        bool has_Z = false;
        float unifiedStrength = 0.0f;  // 可选统一强度
    } _externalParams;
    
    // 渲染参数
    glm::vec3 _meshColor { 0.8f, 0.6f, 0.4f };
    bool _showWireframe = false;
    float _transparency = 1.0f;
    
    // OpenGL 渲染
    Engine::GL::UniqueProgram _program;
    Engine::GL::UniqueRenderFrame _frame;
    Engine::GL::UniqueIndexedRenderItem _meshItem;
    
    // 相机
    Common::OrbitCameraManager _cameraManager;
    Engine::Camera _camera;
    
    // 内部方法
    void LoadSelectedModel();
    void LoadExternalParams();
    void ApplyExternalParams();
    void UpdateVisualization();
    void RenderScene(std::pair<std::uint32_t, std::uint32_t> const size);

public:
    CaseRigid3DSpaces(const std::string& pythonPath = "");
    
    virtual std::string_view const GetName() override { 
        return "Rigid 3D System"; 
    }
    
    virtual void OnSetupPropsUI() override;
    virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
    virtual void OnProcessInput(ImVec2 const & pos) override;
};

} // namespace VCX:: Labs::NeuralPhysicsSubspaces