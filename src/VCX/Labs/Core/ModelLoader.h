#pragma once

#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <regex>

namespace VCX::Labs::NeuralPhysicsSubspaces {

// 模型类型枚举
enum class ModelType {
    FEM,
    Rigid3D,
    Unknown
};

// 检查点范围信息
struct CheckpointRange {
    int minIndex;               // 最小索引（如0）
    int maxIndex;               // 最大索引（如100）
    bool hasFinal;              // 是否有final模型
    std::map<int, std::string> sigmaValues;  // checkpoint index -> sigma string
    
    CheckpointRange() : minIndex(-1), maxIndex(-1), hasFinal(false) {}
};

// 模型配置信息（type + problem + dim + wexp）
struct ModelConfig {
    ModelType type;
    std::string problemName;
    int dim;
    std::string wexp;  // e.g., "wexp1", "wexp0.5"
    CheckpointRange checkpoints;
    
    // 用于map的比较
    bool operator<(const ModelConfig& other) const {
        if (type != other.type) return type < other.type;
        if (problemName != other.problemName) return problemName < other.problemName;
        if (dim != other.dim) return dim < other.dim;
        return wexp < other.wexp;
    }
};

// 模型加载器类（单例模式）
class ModelLoader {
public:
    // 获取单例实例
    static ModelLoader& GetInstance();
    
    // 删除拷贝构造和赋值操作
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    
    // 设置output目录并重新扫描
    void SetOutputDirectory(const std::string& outputDir);
    
    // 扫描output目录中的所有模型
    void ScanModels();
    
    // 获取所有唯一的问题名称
    std::vector<std::string> GetAllProblems() const;
    
    // 获取指定类型和问题的所有维度
    std::vector<int> GetDimsForTypeAndProblem(ModelType type, const std::string& problem) const;
    
    // 获取指定类型、问题和维度的所有wexp值
    std::vector<std::string> GetWexpsForTypeAndProblemDim(ModelType type, const std::string& problem, int dim) const;
    
    // 获取指定类型、问题、维度和wexp的checkpoint范围
    CheckpointRange GetCheckpointRange(ModelType type, const std::string& problem, int dim, const std::string& wexp) const;
    
    // 获取指定checkpoint的sigma值
    std::string GetSigmaForCheckpoint(ModelType type, const std::string& problem, int dim, const std::string& wexp, int index) const;
    
    // 检查特定checkpoint是否存在
    bool CheckpointExists(ModelType type, const std::string& problem, int dim, const std::string& wexp, int index) const;
    
    // 构建模型文件的完整路径（不含扩展名）
    std::string BuildModelPath(ModelType type, const std::string& problem, int dim, const std::string& wexp, int index) const;
    
    // 获取当前output目录
    const std::string& GetOutputDirectory() const { return _outputDir; }
    
private:
    // 私有构造函数
    ModelLoader(const std::string& outputDir = "output");
    ~ModelLoader() = default;
    
    std::string _outputDir;
    std::map<ModelConfig, CheckpointRange> _modelConfigs;  // 存储每个配置的checkpoint范围
    
    // 从文件名解析模型配置
    struct ParsedModel {
        ModelType type;
        std::string problemName;
        int dim;
        std::string wexp;
        int index;  // -1 for final
        std::string sigma;  // e.g., "0", "1e-06", "4.4e-05"
        bool valid;
    };
    ParsedModel ParseFilename(const std::string& filename);
    
    // 构建模型文件名（不含扩展名）
    std::string BuildBaseName(ModelType type, const std::string& problem, int dim, const std::string& wexp, int index) const;
};

} // namespace VCX::Labs::NeuralPhysicsSubspaces
