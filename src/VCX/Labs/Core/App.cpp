#include "Labs/Core/App.h"

namespace VCX::Labs::NeuralPhysicsSubspaces {
    App::App() : 
        _caseFEM(), // 这里可以传入Python路径参数，也可以自动检测
        _caseRigid3D(),
        _ui(Labs::Common::UIOptions { })  {
    }

    void App::OnFrame() {
        _ui.Setup(_cases, _caseId);
    }
}
