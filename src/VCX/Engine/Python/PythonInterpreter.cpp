#include "Engine/Python/PythonInterpreter.h"
#include <spdlog/spdlog.h>

namespace VCX::Engine::Python {

PythonInterpreter& PythonInterpreter::GetInstance() {
    static PythonInterpreter instance;
    return instance;
}

void PythonInterpreter::EnsureInitialized() {
    if (!_guard) {
        spdlog::info("[PythonInterpreter] Initializing global Python interpreter");
        _guard = std::make_unique<pybind11::scoped_interpreter>();
    }
}

} // namespace VCX::Engine::Python
