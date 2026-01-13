#pragma once

#include <vector>

#include "Engine/app.h"
#include "Labs/Core/CaseFEM.h"
#include "Labs/Core/CaseRigid3D.h"
#include "Labs/Core/CaseFEMSubspaces.h"
#include "Labs/Core/CaseRigid3DSubspaces.h"
#include "Labs/Common/UI.h"

namespace VCX::Labs::NeuralPhysicsSubspaces {
    class App : public Engine::IApp {
    private:
        Common::UI             _ui;

        CaseFEM                _caseFEM;
        CaseRigid3D            _caseRigid3D;
        CaseFEMSpaces          _caseFEMSpaces;
        CaseRigid3DSpaces      _caseRigid3DSpaces;

        std::size_t        _caseId = 0;

        std::vector<std::reference_wrapper<Common::ICase>> _cases = {
            _caseFEM, _caseRigid3D, _caseFEMSpaces, _caseRigid3DSpaces
        };

    public:
        App();
        void OnFrame() override;
    };
}
