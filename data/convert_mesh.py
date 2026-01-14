import tetgen
import pyvista as pv
import numpy as np
import os

# 加载你的 obj 文件
mesh_path = 'f:/code/VCI-FinalProject/data/spot.obj'
if not os.path.exists(mesh_path):
    print(f"找不到文件: {mesh_path}")
    exit()

mesh = pv.read(mesh_path)

# 生成四面体网格
tgen = tetgen.TetGen(mesh)

# 使用 switches 参数。'pq1.2' 表示：
# p: 针对 PLC (Piecewise Linear Complex) 进行刨分
# q: 质量剖分 (Quality mesh)，后面跟着的是最小半径-边长比
# 如果提示参数错误，也可以只用 'pq'
tgen.tetrahedralize(switches='pq1.2')

# 获取生成后的节点和单元
nodes = tgen.node
elements = tgen.elem

# 保存为 FEM 代码需要的格式 (.node 和 .ele)
def save_tet_mesh(name, nodes, elements):
    # 保存 .node 文件
    with open(f"{name}.node", "w") as f:
        # 格式: [节点数] [维度(3)] [属性数(0)] [边界标志(0)]
        f.write(f"{len(nodes)} 3 0 0\n")
        for i, v in enumerate(nodes):
            f.write(f"{i} {v[0]} {v[1]} {v[2]}\n")
            
    # 保存 .ele 文件
    with open(f"{name}.ele", "w") as f:
        # 格式: [单元数] [节点数/单元(4)] [属性数(0)]
        f.write(f"{len(elements)} 4 0\n")
        for i, e in enumerate(elements):
            f.write(f"{i} {e[0]} {e[1]} {e[2]} {e[3]}\n")

output_base = 'f:/code/VCI-FinalProject/data/spot'
save_tet_mesh(output_base, nodes, elements)
print(f"转换完成！已生成 {output_base}.node 和 {output_base}.ele")