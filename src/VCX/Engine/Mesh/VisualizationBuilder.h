#pragma once

#include "MeshDataLoader.h"
#include <glm/glm.hpp>
#include <vector>

namespace VCX::Engine::Mesh {

// 可视化输出数据
struct VisualizationBody {
    std::vector<glm::vec3> vertices;
    std::vector<glm::uvec3> faces;
    std::vector<glm::vec3> colors;
};

using MeshVisualizationData = std::vector<VisualizationBody>;

// 可视化生成器
class VisualizationBuilder {
public:
    // FEM可视化：输入自由度向量q（全空间），输出可视化网格
    static MeshVisualizationData BuildFEMVisualization(
        const FEMMeshData& mesh,
        const std::vector<double>& q
    );
    
    // Rigid3D可视化：输入完整状态（包含固定刚体），输出多个刚体网格
    static MeshVisualizationData BuildRigid3DVisualization(
        const Rigid3DMeshData& mesh,
        const std::vector<double>& q_full  // 所有刚体的4x3仿射矩阵展平
    );
    
private:
    // 根据region分配颜色
    static glm::vec3 GetRegionColor(int region);
    
    // HSV转RGB（用于刚体配色）
    static glm::vec3 HSVtoRGB(float h, float s, float v);
    
    // 为三角形找到所属四面体（用于region映射）
    static int FindTetForTriangle(
        const glm::uvec3& tri,
        const std::vector<glm::uvec4>& tets
    );
};

} // namespace VCX::Engine::Mesh
