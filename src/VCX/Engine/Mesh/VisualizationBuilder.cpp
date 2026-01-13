#include "VisualizationBuilder.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <set>

namespace VCX::Engine::Mesh {

// ==================== FEM可视化生成 ====================

MeshVisualizationData VisualizationBuilder::BuildFEMVisualization(
    const FEMMeshData& mesh,
    const std::vector<double>& q
) {
    MeshVisualizationData result;
    VisualizationBody body;
    
    // 1. 从q重建当前顶点位置
    int num_verts = mesh.Vrest.size();
    std::vector<glm::vec3> current_pos(num_verts);
    const size_t expected_size = static_cast<size_t>(num_verts * mesh.pos_dim);
    const size_t q_size = q.size();
    static int mismatch_log_count = 0; 

    if (q_size == expected_size) {
        // q直接是所有顶点的位置
        for (int i = 0; i < num_verts; i++) {
            if (mesh.pos_dim == 2) {
                current_pos[i] = glm::vec3(q[i * 2], q[i * 2 + 1], 0.0f);
            } else {
                current_pos[i] = glm::vec3(q[i * 3], q[i * 3 + 1], q[i * 3 + 2]);
            }
        }
    } else if (!mesh.unfixed_inds.empty() && q_size == mesh.unfixed_inds.size() * mesh.pos_dim) {
        // q只包含自由顶点，填回全尺寸
        current_pos = mesh.Vrest; // 先放入rest，随后覆盖自由顶点
        for (size_t k = 0; k < mesh.unfixed_inds.size(); k++) {
            int vid = mesh.unfixed_inds[k];
            if (vid < 0 || vid >= num_verts) continue;
            if (mesh.pos_dim == 2) {
                current_pos[vid] = glm::vec3(q[k * 2], q[k * 2 + 1], 0.0f);
            } else {
                current_pos[vid] = glm::vec3(q[k * 3], q[k * 3 + 1], q[k * 3 + 2]);
            }
        }
    } else {
        if (mismatch_log_count < 5) {
            spdlog::warn("[VisualizationBuilder] q size mismatch (got {}, expected full {}), using rest positions", q_size, expected_size);
            mismatch_log_count++;
        }
        current_pos = mesh.Vrest;
    }
    
    body.vertices = current_pos;
    
    // 2. 选择faces
    if (mesh.pos_dim == 2) {
        // 2D：直接用三角形
        body.faces = mesh.E_tri;
    } else {
        // 3D：用边界三角形
        if (!mesh.boundary_triangles.empty()) {
            body.faces = mesh.boundary_triangles;
        } else {
            // 如果没有预计算，用所有三角形（或tet的某个面）
            spdlog::warn("[VisualizationBuilder] No boundary triangles, using E_tri");
            body.faces = mesh.E_tri;
        }
    }
    
    // 3. 生成颜色
    body.colors.reserve(body.faces.size());
    
    if (mesh.regions.empty()) {
        // 无region：统一颜色
        glm::vec3 default_color(0.3f, 0.5f, 0.9f);
        for (size_t i = 0; i < body.faces.size(); i++) {
            body.colors.push_back(default_color);
        }
    } else {
        // 有region：需要映射
        if (mesh.pos_dim == 2 || mesh.regions.size() == body.faces.size()) {
            // regions直接对应faces
            for (size_t i = 0; i < body.faces.size(); i++) {
                int region = (i < mesh.regions.size()) ? mesh.regions[i] : 0;
                body.colors.push_back(GetRegionColor(region));
            }
        } else {
            // regions对应tets，需要为每个boundary face找到所属tet
            for (const auto& face : body.faces) {
                int tet_idx = FindTetForTriangle(face, mesh.E_tet);
                int region = (tet_idx >= 0 && tet_idx < mesh.regions.size()) 
                             ? mesh.regions[tet_idx] : 0;
                body.colors.push_back(GetRegionColor(region));
            }
        }
    }
    
    result.push_back(body);
    
    // 4. 添加额外几何体（bistable的endcaps）
    if (mesh.problem_name == "bistable" && mesh.pos_dim == 2) {
        VisualizationBody extra;
        
        float s = 0.4f, w = 0.07f, h = 0.1f;
        extra.vertices = {
            glm::vec3(-s-w, -h, 0), glm::vec3(-s, -h, 0),
            glm::vec3(-s, h, 0),    glm::vec3(-s-w, h, 0),
            glm::vec3(s, -h, 0),    glm::vec3(s+w, -h, 0),
            glm::vec3(s+w, h, 0),   glm::vec3(s, h, 0)
        };
        
        // 两个四边形，拆成4个三角形
        extra.faces = {
            glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3),  // 左侧
            glm::uvec3(4, 5, 6), glm::uvec3(4, 6, 7)   // 右侧
        };
        
        glm::vec3 gray(0.7f);
        extra.colors = {gray, gray, gray, gray};
        
        result.push_back(extra);
    }
    
    return result;
}

// ==================== Rigid3D可视化生成 ====================

MeshVisualizationData VisualizationBuilder::BuildRigid3DVisualization(
    const Rigid3DMeshData& mesh,
    const std::vector<double>& q_full
) {
    MeshVisualizationData result;
    
    int n_bodies = mesh.bodies.size();
    int expected_size = n_bodies * 12; // 每个body 4x3=12个数
    
    if (q_full.size() != expected_size) {
        spdlog::error("[VisualizationBuilder] Rigid3D state size mismatch: {} vs expected {}", 
                      q_full.size(), expected_size);
        return result;
    }
    
    // 将q_full解析为仿射矩阵（4x3）
    std::vector<Eigen::Matrix<double, 4, 3>> xr(n_bodies);
    for (int bid = 0; bid < n_bodies; bid++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                xr[bid](i, j) = q_full[bid * 12 + i * 3 + j];
            }
        }
    }
    
    // 为每个body生成变换后的网格
    for (int bid = 0; bid < n_bodies; bid++) {
        const auto& body_data = mesh.bodies[bid];
        VisualizationBody body;
        
        // 顶点变换：v' = W * xr
        Eigen::MatrixXd transformed = body_data.W * xr[bid];
        
        body.vertices.reserve(transformed.rows());
        for (int i = 0; i < transformed.rows(); i++) {
            body.vertices.push_back(glm::vec3(
                transformed(i, 0),
                transformed(i, 1),
                transformed(i, 2)
            ));
        }
        
        // faces直接复制
        body.faces = body_data.f;
        
        // 颜色：用HSV色轮给每个body分配颜色
        float hue = (float)bid / (float)n_bodies * 360.0f;
        glm::vec3 body_color = HSVtoRGB(hue, 0.7f, 0.7f);
        
        body.colors.resize(body.faces.size(), body_color);
        
        result.push_back(body);
    }
    
    return result;
}

// ==================== 辅助函数 ====================

glm::vec3 VisualizationBuilder::GetRegionColor(int region) {
    switch (region) {
        case 0: return glm::vec3(0.3f, 0.5f, 0.9f);  // 蓝色
        case 1: return glm::vec3(0.4f, 0.8f, 0.4f);  // 绿色
        case 2: return glm::vec3(0.9f, 0.4f, 0.3f);  // 红色
        default: return glm::vec3(0.7f, 0.7f, 0.7f); // 灰色
    }
}

glm::vec3 VisualizationBuilder::HSVtoRGB(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    
    glm::vec3 rgb;
    if (h < 60.0f)       rgb = glm::vec3(c, x, 0);
    else if (h < 120.0f) rgb = glm::vec3(x, c, 0);
    else if (h < 180.0f) rgb = glm::vec3(0, c, x);
    else if (h < 240.0f) rgb = glm::vec3(0, x, c);
    else if (h < 300.0f) rgb = glm::vec3(x, 0, c);
    else                 rgb = glm::vec3(c, 0, x);
    
    return rgb + glm::vec3(m);
}

int VisualizationBuilder::FindTetForTriangle(
    const glm::uvec3& tri,
    const std::vector<glm::uvec4>& tets
) {
    std::set<uint32_t> tri_verts = {tri.x, tri.y, tri.z};
    
    for (size_t t = 0; t < tets.size(); t++) {
        const auto& tet = tets[t];
        std::set<uint32_t> tet_verts = {tet.x, tet.y, tet.z, tet.w};
        
        // 检查tri的三个顶点是否都在tet中
        bool all_in = true;
        for (uint32_t v : tri_verts) {
            if (tet_verts.find(v) == tet_verts.end()) {
                all_in = false;
                break;
            }
        }
        
        if (all_in) return t;
    }
    
    return -1; // 未找到
}

} // namespace VCX::Engine::Mesh
