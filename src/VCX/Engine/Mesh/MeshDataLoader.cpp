#include "MeshDataLoader.h"
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace VCX::Engine::Mesh {

// ==================== FEM网格加载 ====================

FEMMeshData MeshDataLoader::LoadFEMMesh(const std::string& problem_name) {
    FEMMeshData mesh;
    mesh.problem_name = problem_name;
    
    if (problem_name == "bistable") {
        LoadTriMesh("./data/longerCantileverP2", mesh);
        mesh.pos_dim = 2;
    } else if (problem_name == "load3d") {
        LoadTetMeshMEDIT("./data/beam365.mesh", mesh);
        mesh.pos_dim = 3;
    } else if (problem_name == "heterobeam") {
        LoadTetMesh("./data/heterobeam", mesh);
        mesh.pos_dim = 3;
    } else {
        spdlog::error("[MeshDataLoader] Unknown FEM problem: {}", problem_name);
        mesh.pos_dim = 2;
    }
    
    return mesh;
}

void MeshDataLoader::LoadTriMesh(const std::string& file_root, FEMMeshData& mesh) {
    // Triangle格式简化的手动解析
    std::ifstream ele_file(file_root + ".ele");
    std::ifstream node_file(file_root + ".node");
    
    if (!ele_file || !node_file) {
        spdlog::error("[MeshDataLoader] Cannot open {} files", file_root);
        return;
    }
    
    auto skip_comments = [](std::ifstream& file, std::string& line) {
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') return true;
        }
        return false;
    };
    
    // 读取顶点
    std::string line;
    skip_comments(node_file, line);
    std::istringstream node_header(line);
    int n_vert, n_dim, n_attrib, n_bmark;
    node_header >> n_vert >> n_dim >> n_attrib >> n_bmark;
    
    std::vector<glm::vec2> verts_2d;
    verts_2d.reserve(n_vert);
    int first_node_idx = -1;
    
    for (int i = 0; i < n_vert; i++) {
        if (!std::getline(node_file, line) || line.empty() || line[0] == '#') { i--; continue; }
        std::istringstream iss(line);
        int idx; double x, y;
        iss >> idx >> x >> y;
        if (first_node_idx == -1) first_node_idx = idx;
        verts_2d.push_back(glm::vec2(x, y));
    }
    
    // 读取三角形
    skip_comments(ele_file, line);
    std::istringstream ele_header(line);
    int n_tri, n_nodes, n_attrib_e;
    ele_header >> n_tri >> n_nodes >> n_attrib_e;
    
    mesh.E_tri.reserve(n_tri);
    if (n_attrib_e > 0) mesh.regions.reserve(n_tri);
    int first_ele_idx = -1;
    
    for (int i = 0; i < n_tri; i++) {
        if (!std::getline(ele_file, line) || line.empty() || line[0] == '#') { i--; continue; }
        std::istringstream iss(line);
        int idx, v0, v1, v2;
        iss >> idx >> v0 >> v1 >> v2;
        if (first_ele_idx == -1) first_ele_idx = idx;
        mesh.E_tri.push_back(glm::uvec3(v0 - first_node_idx, v1 - first_node_idx, v2 - first_node_idx));
        
        if (n_attrib_e > 0) {
            int region;
            iss >> region;
            mesh.regions.push_back(region);
        }
    }
    
    // 归一化
    glm::vec2 vmin(1e10), vmax(-1e10);
    for (const auto& v : verts_2d) {
        vmin = glm::min(vmin, v);
        vmax = glm::max(vmax, v);
    }
    glm::vec2 center = (vmin + vmax) * 0.5f;
    float scale = std::max(vmax.x - vmin.x, vmax.y - vmin.y);
    
    mesh.Vrest.reserve(verts_2d.size());
    for (const auto& v : verts_2d) {
        glm::vec2 normalized = (v - center) / scale;
        mesh.Vrest.push_back(glm::vec3(normalized.x, normalized.y, 0.0f));
    }
    
    spdlog::info("[MeshDataLoader] Loaded 2D mesh: {} verts, {} triangles", 
                 mesh.Vrest.size(), mesh.E_tri.size());
}

void MeshDataLoader::LoadTetMesh(const std::string& file_root, FEMMeshData& mesh) {
    // Triangle格式没有简化的手动解析
    std::ifstream ele_file(file_root + ".ele");
    std::ifstream node_file(file_root + ".node");
    
    if (!ele_file || !node_file) {
        spdlog::error("[MeshDataLoader] Cannot open {} files", file_root);
        return;
    }
    
    auto skip_comments = [](std::ifstream& file, std::string& line) {
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') return true;
        }
        return false;
    };
    
    // 读取顶点
    std::string line;
    skip_comments(node_file, line);
    std::istringstream node_header(line);
    int n_vert, n_dim, n_attrib, n_bmark;
    node_header >> n_vert >> n_dim >> n_attrib >> n_bmark;
    
    std::vector<glm::vec3> verts_3d;
    verts_3d.reserve(n_vert);
    int first_node_idx = -1;
    
    for (int i = 0; i < n_vert; i++) {
        if (!std::getline(node_file, line) || line.empty() || line[0] == '#') { i--; continue; }
        std::istringstream iss(line);
        int idx; double x, y, z;
        iss >> idx >> x >> y >> z;
        if (first_node_idx == -1) first_node_idx = idx;
        verts_3d.push_back(glm::vec3(x, y, z));
    }
    
    // 读取四面体
    skip_comments(ele_file, line);
    std::istringstream ele_header(line);
    int n_tet, n_nodes, n_attrib_e;
    ele_header >> n_tet >> n_nodes >> n_attrib_e;
    
    mesh.E_tet.reserve(n_tet);
    if (n_attrib_e > 0) mesh.regions.reserve(n_tet);
    int first_ele_idx = -1;
    
    for (int i = 0; i < n_tet; i++) {
        if (!std::getline(ele_file, line) || line.empty() || line[0] == '#') { i--; continue; }
        std::istringstream iss(line);
        int idx, v0, v1, v2, v3;
        iss >> idx >> v0 >> v1 >> v2 >> v3;
        if (first_ele_idx == -1) first_ele_idx = idx;
        mesh.E_tet.push_back(glm::uvec4(
            v0 - first_node_idx, v1 - first_node_idx, 
            v2 - first_node_idx, v3 - first_node_idx
        ));
        
        if (n_attrib_e > 0) {
            int region;
            iss >> region;
            mesh.regions.push_back(region);
        }
    }
    
    // 归一化
    glm::vec3 vmin(1e10), vmax(-1e10);
    for (const auto& v : verts_3d) {
        vmin = glm::min(vmin, v);
        vmax = glm::max(vmax, v);
    }
    glm::vec3 center = (vmin + vmax) * 0.5f;
    float scale = std::max({vmax.x - vmin.x, vmax.y - vmin.y, vmax.z - vmin.z});
    
    mesh.Vrest.reserve(verts_3d.size());
    for (const auto& v : verts_3d) {
        mesh.Vrest.push_back((v - center) / scale);
    }
    
    // 使用igl提取边界三角形
    Eigen::MatrixXi T(mesh.E_tet.size(), 4);
    for (size_t i = 0; i < mesh.E_tet.size(); i++) {
        T(i, 0) = mesh.E_tet[i].x;
        T(i, 1) = mesh.E_tet[i].y;
        T(i, 2) = mesh.E_tet[i].z;
        T(i, 3) = mesh.E_tet[i].w;
    }
    Eigen::MatrixXi boundary_faces;
    igl::boundary_facets(T, boundary_faces);
    
    mesh.boundary_triangles.reserve(boundary_faces.rows());
    for (int i = 0; i < boundary_faces.rows(); i++) {
        mesh.boundary_triangles.push_back(glm::uvec3(
            boundary_faces(i, 0), boundary_faces(i, 1), boundary_faces(i, 2)
        ));
    }
    
    spdlog::info("[MeshDataLoader] Loaded 3D mesh: {} verts, {} tets, {} boundary faces", 
                 mesh.Vrest.size(), mesh.E_tet.size(), mesh.boundary_triangles.size());
}

void MeshDataLoader::LoadTetMeshMEDIT(const std::string& mesh_path, FEMMeshData& mesh) {
    // 使用libigl读取MEDIT格式
    Eigen::MatrixXd V;
    Eigen::MatrixXi T, F;
    
    if (!igl::readMESH(mesh_path, V, T, F)) {
        spdlog::error("[MeshDataLoader] Failed to read mesh file: {}", mesh_path);
        return;
    }

    if (V.rows() == 0 || T.rows() == 0) {
        spdlog::error("[MeshDataLoader] Empty mesh in {}", mesh_path);
        return;
    }

    // 归一化顶点（与python侧保持一致）
    Eigen::Vector3d vmin = V.colwise().minCoeff();
    Eigen::Vector3d vmax = V.colwise().maxCoeff();
    Eigen::Vector3d center = (vmin + vmax) * 0.5;
    Eigen::Vector3d range = vmax - vmin;
    double scale = range.maxCoeff();
    if (scale <= 0.0) scale = 1.0;

    // 归一化并转换为glm格式
    mesh.Vrest.reserve(V.rows());
    for (int i = 0; i < V.rows(); i++) {
        Eigen::Vector3d normalized = (V.row(i).transpose() - center) / scale;
        mesh.Vrest.push_back(glm::vec3(normalized.x(), normalized.y(), normalized.z()));
    }

    // 转换四面体
    mesh.E_tet.reserve(T.rows());
    for (int i = 0; i < T.rows(); i++) {
        mesh.E_tet.push_back(glm::uvec4(T(i, 0), T(i, 1), T(i, 2), T(i, 3)));
    }

    // 使用igl提取边界三角形
    Eigen::MatrixXi boundary_faces;
    igl::boundary_facets(T, boundary_faces);
    
    mesh.boundary_triangles.reserve(boundary_faces.rows());
    for (int i = 0; i < boundary_faces.rows(); i++) {
        mesh.boundary_triangles.push_back(glm::uvec3(
            boundary_faces(i, 0), 
            boundary_faces(i, 1), 
            boundary_faces(i, 2)
        ));
    }

    spdlog::info("[MeshDataLoader] Loaded .mesh tet with libigl: {} verts, {} tets, {} boundary faces", 
                 mesh.Vrest.size(), mesh.E_tet.size(), mesh.boundary_triangles.size());
}

std::vector<glm::uvec3> MeshDataLoader::ExtractBoundaryTriangles(const std::vector<glm::uvec4>& tets) {
    // 用哈希表统计每个三角形出现次数，只出现一次的是边界
    std::map<std::set<uint32_t>, uint32_t> face_count;
    
    auto add_face = [&](uint32_t v0, uint32_t v1, uint32_t v2) {
        std::set<uint32_t> key = {v0, v1, v2};
        face_count[key]++;
    };
    
    for (const auto& tet : tets) {
        // 四面体的4个面
        add_face(tet.x, tet.y, tet.z);
        add_face(tet.x, tet.y, tet.w);
        add_face(tet.x, tet.z, tet.w);
        add_face(tet.y, tet.z, tet.w);
    }
    
    std::vector<glm::uvec3> boundary;
    for (const auto& [verts, count] : face_count) {
        if (count == 1) {
            auto it = verts.begin();
            uint32_t v0 = *it++; 
            uint32_t v1 = *it++; 
            uint32_t v2 = *it;
            boundary.push_back(glm::uvec3(v0, v1, v2));
        }
    }
    
    return boundary;
}

// ==================== Rigid3D网格加载 ====================

Rigid3DMeshData MeshDataLoader::LoadRigid3DMesh(const std::string& problem_name) {
    Rigid3DMeshData mesh;
    mesh.problem_name = problem_name;
    
    if (problem_name == "klann") {
        mesh.bodies.push_back(LoadRigidBody("./data/klann-red.obj", 1000, 1.0));
        mesh.bodies.push_back(LoadRigidBody("./data/klann-purple.obj", 1000, 1.0));
        mesh.bodies.push_back(LoadRigidBody("./data/klann-brown.obj", 1000, 1.0));
        mesh.bodies.push_back(LoadRigidBody("./data/klann-distal.obj", 1000, 1.0));
        mesh.bodies.push_back(LoadRigidBody("./data/klann-top.obj", 1000, 1.0));
        mesh.num_fixed_bodies = 0;
        
    } else if (problem_name == "stewart") {
        double scale = 5.0;
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-base.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-arm1.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-arm2.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-arm3.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-arm4.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-arm5.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-arm6.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-strut1.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-strut2.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-strut3.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-strut4.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-strut5.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-strut6.obj", 1000, scale));
        mesh.bodies.push_back(LoadRigidBody("./data/stewart-top.obj", 1000, scale));
        mesh.num_fixed_bodies = 1;
        
    } else {
        spdlog::error("[MeshDataLoader] Unknown Rigid3D problem: {}", problem_name);
    }
    
    return mesh;
}

RigidBodyData MeshDataLoader::LoadRigidBody(const std::string& obj_path, double density, double scale) {
    RigidBodyData body;
    
    // 使用libigl读取OBJ文件（更可靠）
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    
    if (!igl::read_triangle_mesh(obj_path, V, F)) {
        spdlog::error("[MeshDataLoader] Failed to read OBJ file: {}", obj_path);
        return body;
    }
    
    if (V.rows() == 0 || F.rows() == 0) {
        spdlog::error("[MeshDataLoader] Empty mesh in {}", obj_path);
        return body;
    }
    
    // 应用缩放
    V *= scale;
    
    // 使用igl.massmatrix计算面积加权质心（与Python端完全一致）
    Eigen::SparseMatrix<double> M;
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
    
    // M是稀疏对角矩阵，每个顶点对应的质量（面积权重）
    // Python端使用np.nan_to_num处理NaN值
    Eigen::VectorXd masses = M.diagonal();
    for (int i = 0; i < masses.size(); i++) {
        if (std::isnan(masses(i)) || std::isinf(masses(i))) {
            masses(i) = 0.0;
        }
    }
    
    double total_mass = masses.sum();
    
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    if (total_mass > 0.0) {
        for (int i = 0; i < V.rows(); i++) {
            center += masses(i) * V.row(i).transpose();
        }
        center /= total_mass;
    }
    
    // 转换为glm格式并去中心化
    body.v.reserve(V.rows());
    for (int i = 0; i < V.rows(); i++) {
        glm::vec3 vert(V(i, 0), V(i, 1), V(i, 2));
        glm::vec3 centered = vert - glm::vec3(center.x(), center.y(), center.z());
        body.v.push_back(centered);
    }
    
    // 转换面
    body.f.reserve(F.rows());
    for (int i = 0; i < F.rows(); i++) {
        body.f.push_back(glm::uvec3(F(i, 0), F(i, 1), F(i, 2)));
    }
    
    // 构建W矩阵 [v, 1]
    body.W.resize(body.v.size(), 4);
    for (size_t i = 0; i < body.v.size(); i++) {
        body.W(i, 0) = body.v[i].x;
        body.W(i, 1) = body.v[i].y;
        body.W(i, 2) = body.v[i].z;
        body.W(i, 3) = 1.0;
    }
    
    spdlog::info("[MeshDataLoader] Loaded rigid body from {}: {} verts, {} faces", 
                 obj_path, body.v.size(), body.f.size());
    
    return body;
}

} // namespace VCX::Engine::Mesh
