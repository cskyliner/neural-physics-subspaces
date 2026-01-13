#include "ModelLoader.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <set>
#include <filesystem>

namespace VCX::Labs::NeuralPhysicsSubspaces {

// 单例实例获取
ModelLoader& ModelLoader::GetInstance() {
    static ModelLoader instance;
    return instance;
}

// 私有构造函数
ModelLoader::ModelLoader(const std::string& outputDir)
    : _outputDir(outputDir) {
    ScanModels();
}

// 设置output目录
void ModelLoader::SetOutputDirectory(const std::string& outputDir) {
    if (_outputDir != outputDir) {
        _outputDir = outputDir;
        ScanModels();
        spdlog::info("[ModelLoader] Output directory changed to: {}", _outputDir);
    }
}


void ModelLoader::ScanModels() {
    _modelConfigs.clear();
    
    namespace fs = std::filesystem;
    
    if (!fs::exists(_outputDir) || !fs::is_directory(_outputDir)) {
        spdlog::warn("[ModelLoader] Output directory not found: {}", _outputDir);
        return;
    }
    
    // 临时存储：每个配置对应的所有索引和sigma值
    std::map<ModelConfig, std::set<int>> tempIndices;
    std::map<ModelConfig, std::map<int, std::string>> tempSigmas;
    
    for (const auto& entry : fs::directory_iterator(_outputDir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filename = entry.path().filename().string();
        
        // 只处理.json文件
        if (!filename.ends_with(".json")) continue;
        
        // 移除扩展名
        std::string baseName = filename.substr(0, filename.length() - 5);
        
        // 解析文件名
        ParsedModel parsed = ParseFilename(baseName);
        if (!parsed.valid) continue;
        
        // 构建配置key
        ModelConfig config;
        config.type = parsed.type;
        config.problemName = parsed.problemName;
        config.dim = parsed.dim;
        config.wexp = parsed.wexp;
        
        // 添加索引和sigma值
        tempIndices[config].insert(parsed.index);
        if (parsed.index >= 0 && !parsed.sigma.empty()) {
            // 存储checkpoint的sigma值
            tempSigmas[config][parsed.index] = parsed.sigma;
        }
    }
    
    // 将索引集合转换为范围
    for (auto& [config, indices] : tempIndices) {
        CheckpointRange range;
        range.hasFinal = false;
        range.minIndex = -1;
        range.maxIndex = -1;
        
        for (int idx : indices) {
            if (idx == -1) {
                range.hasFinal = true;
            } else {
                if (range.minIndex == -1 || idx < range.minIndex) {
                    range.minIndex = idx;
                }
                if (range.maxIndex == -1 || idx > range.maxIndex) {
                    range.maxIndex = idx;
                }
            }
        }
        
        // 复制sigma值
        if (tempSigmas.find(config) != tempSigmas.end()) {
            range.sigmaValues = tempSigmas[config];
        }
        
        ModelConfig configWithRange = config;
        configWithRange.checkpoints = range;
        _modelConfigs[configWithRange] = range;
    }
    
    spdlog::info("[ModelLoader] Found {} model configurations in {}", _modelConfigs.size(), _outputDir);
}

ModelLoader::ParsedModel ModelLoader::ParseFilename(const std::string& filename) {
    ParsedModel result;
    result.valid = false;
    result.type = ModelType::Unknown;
    result.dim = -1;
    result.index = -1;
    
    // 确定类型
    if (filename.find("_fem_") != std::string::npos) {
        result.type = ModelType::FEM;
    } else if (filename.find("_rigid3d_") != std::string::npos) {
        result.type = ModelType::Rigid3D;
    } else {
        return result;
    }
    
    // 解析问题名称: neural_subspace_(fem|rigid3d)_<problem>_
    std::regex problemRegex("neural_subspace_(?:fem|rigid3d)_([^_]+)_");
    std::smatch match;
    if (std::regex_search(filename, match, problemRegex)) {
        result.problemName = match[1].str();
    } else {
        return result;
    }
    
    // 解析维度: _dim(\d+)_
    std::regex dimRegex("_dim(\\d+)_");
    if (std::regex_search(filename, match, dimRegex)) {
        result.dim = std::stoi(match[1].str());
    } else {
        return result;
    }
    
    // 解析wexp: _wexp([0-9.]+)_
    std::regex wexpRegex("_wexp([0-9.]+)_");
    if (std::regex_search(filename, match, wexpRegex)) {
        result.wexp = "wexp" + match[1].str();
    } else {
        return result;
    }
    
    // 判断是final还是checkpoint
    if (filename.find("_final") != std::string::npos) {
        result.index = -1;  // -1表示final
        result.sigma = "";  // final没有sigma
    } else {
        // 解析checkpoint索引: _save(\d{4})_
        std::regex saveRegex("_save(\\d{4})_");
        if (std::regex_search(filename, match, saveRegex)) {
            result.index = std::stoi(match[1].str());
        } else {
            return result;
        }
        
        // 解析sigma值: _sigma([0-9.e-]+)
        std::regex sigmaRegex("_sigma([0-9.e-]+)");
        if (std::regex_search(filename, match, sigmaRegex)) {
            result.sigma = match[1].str();
        } else {
            return result;
        }
    }
    
    result.valid = true;
    return result;
}

std::vector<std::string> ModelLoader::GetAllProblems() const {
    std::set<std::string> problems;
    for (const auto& [config, range] : _modelConfigs) {
        if (!config.problemName.empty()) {
            problems.insert(config.problemName);
        }
    }
    return std::vector<std::string>(problems.begin(), problems.end());
}

std::vector<int> ModelLoader::GetDimsForTypeAndProblem(ModelType type, const std::string& problem) const {
    std::set<int> dims;
    for (const auto& [config, range] : _modelConfigs) {
        if (config.type == type && config.problemName == problem && config.dim > 0) {
            dims.insert(config.dim);
        }
    }
    return std::vector<int>(dims.begin(), dims.end());
}

std::vector<std::string> ModelLoader::GetWexpsForTypeAndProblemDim(ModelType type, const std::string& problem, int dim) const {
    std::set<std::string> wexps;
    for (const auto& [config, range] : _modelConfigs) {
        if (config.type == type && config.problemName == problem && config.dim == dim && !config.wexp.empty()) {
            wexps.insert(config.wexp);
        }
    }
    return std::vector<std::string>(wexps.begin(), wexps.end());
}

CheckpointRange ModelLoader::GetCheckpointRange(ModelType type, const std::string& problem, int dim, const std::string& wexp) const {
    ModelConfig key;
    key.type = type;
    key.problemName = problem;
    key.dim = dim;
    key.wexp = wexp;
    
    auto it = _modelConfigs.find(key);
    if (it != _modelConfigs.end()) {
        return it->second;
    }
    
    return CheckpointRange();  // 返回空范围
}

std::string ModelLoader::GetSigmaForCheckpoint(ModelType type, const std::string& problem, int dim, const std::string& wexp, int index) const {
    CheckpointRange range = GetCheckpointRange(type, problem, dim, wexp);
    auto it = range.sigmaValues.find(index);
    if (it != range.sigmaValues.end()) {
        return it->second;
    }
    // Fallback: 如果找不到sigma值，使用旧的计算方式
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.3f", index * 0.001f);
    return std::string(buffer);
}

bool ModelLoader::CheckpointExists(ModelType type, const std::string& problem, int dim, const std::string& wexp, int index) const {
    namespace fs = std::filesystem;
    
    std::string baseName = BuildBaseName(type, problem, dim, wexp, index);
    std::string jsonPath = (fs::path(_outputDir) / (baseName + ".json")).string();
    
    return fs::exists(jsonPath);
}

std::string ModelLoader::BuildModelPath(ModelType type, const std::string& problem, int dim, const std::string& wexp, int index) const {
    namespace fs = std::filesystem;
    std::string baseName = BuildBaseName(type, problem, dim, wexp, index);
    return (fs::path(_outputDir) / baseName).string();
}

std::string ModelLoader::BuildBaseName(ModelType type, const std::string& problem, int dim, const std::string& wexp, int index) const {
    std::string typeStr = (type == ModelType::FEM) ? "fem" : "rigid3d";
    
    char buffer[256];
    if (index == -1) {
        // Final模型
        snprintf(buffer, sizeof(buffer),
                "neural_subspace_%s_%s_normal_dim%d_%s_final",
                typeStr.c_str(), problem.c_str(), dim, wexp.c_str());
    } else {
        // Checkpoint模型 - 使用实际的sigma值
        std::string sigma = GetSigmaForCheckpoint(type, problem, dim, wexp, index);
        snprintf(buffer, sizeof(buffer),
                "neural_subspace_%s_%s_normal_dim%d_%s_save%04d_sigma%s",
                typeStr.c_str(), problem.c_str(), dim, wexp.c_str(), index, sigma.c_str());
    }
    
    return std::string(buffer);
}

} // namespace VCX::Labs::NeuralPhysicsSubspaces
