#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <igl/read_triangle_mesh.h>
#include <igl/readMESH.h>
#include <igl/boundary_facets.h>
#include <igl/massmatrix.h>

namespace VCX::Engine::Mesh {

// FEM网格数据
struct FEMMeshData {
    std::vector<glm::vec3> Vrest;           // 初始顶点位置（统一用vec3，2D时z=0）
    std::vector<glm::uvec3> E_tri;          // 三角形单元（用于2D或3D边界）
    std::vector<glm::uvec4> E_tet;          // 四面体单元（仅3D）
    std::vector<glm::uvec3> boundary_triangles; // 四面体的边界三角形
    std::vector<int> regions;                // 材料区域标记（与E对应）
    
    int pos_dim;                             // 2 or 3
    std::string problem_name;
    
    // 固定顶点索引与值（若系统有fixed entries）
    std::vector<int> fixed_inds;
    std::vector<int> unfixed_inds;
    std::vector<double> fixed_values;
};

// Rigid3D刚体网格数据
struct RigidBodyData {
    std::vector<glm::vec3> v;               // 初始局部顶点（已去质心）
    std::vector<glm::uvec3> f;              // 三角形面
    Eigen::MatrixXd W;                       // [v, 1] 扩展矩阵（用于仿射变换）
};

struct Rigid3DMeshData {
    std::vector<RigidBodyData> bodies;
    int num_fixed_bodies;                    // 前N个body是固定的
    std::string problem_name;
};

// 网格加载器
class MeshDataLoader {
public:
    // 加载FEM网格（.ele/.node或.mesh）
    static FEMMeshData LoadFEMMesh(const std::string& problem_name);
    
    // 加载Rigid3D刚体集合
    static Rigid3DMeshData LoadRigid3DMesh(const std::string& problem_name);
    
private:
    // 辅助函数：从.ele/.node加载三角形网格
    static void LoadTriMesh(const std::string& file_root, FEMMeshData& mesh);
    
    // 辅助函数：从.ele/.node加载四面体网格
    static void LoadTetMesh(const std::string& file_root, FEMMeshData& mesh);

    // 辅助函数：从 .mesh (MEDIT) 加载四面体网格
    static void LoadTetMeshMEDIT(const std::string& mesh_path, FEMMeshData& mesh);
    
    // 辅助函数：从.obj加载FEM三角形网格
    static void LoadObjMesh(const std::string& obj_path, FEMMeshData& mesh);
    
    // 辅助函数：提取四面体边界三角形
    static std::vector<glm::uvec3> ExtractBoundaryTriangles(const std::vector<glm::uvec4>& tets);
    
    // 辅助函数：从.obj加载刚体
    static RigidBodyData LoadRigidBody(const std::string& obj_path, double density, double scale);
};

} // namespace VCX::Engine::Mesh
