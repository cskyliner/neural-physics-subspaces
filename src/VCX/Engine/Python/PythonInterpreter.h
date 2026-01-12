#pragma once

#include <pybind11/embed.h>
#include <memory>

namespace VCX::Engine::Python {

// 全局 Python 解释器单例管理
// 确保整个程序只有一个 Python 解释器实例
class PythonInterpreter {
public:
    // 获取全局解释器单例
    static PythonInterpreter& GetInstance();
    
    // 禁止拷贝和赋值
    PythonInterpreter(const PythonInterpreter&) = delete;
    PythonInterpreter& operator=(const PythonInterpreter&) = delete;
    
    // 确保解释器已初始化
    void EnsureInitialized();
    
private:
    PythonInterpreter() = default;
    ~PythonInterpreter() = default;
    
    std::unique_ptr<pybind11::scoped_interpreter> _guard;
};

} // namespace VCX::Engine::Python
