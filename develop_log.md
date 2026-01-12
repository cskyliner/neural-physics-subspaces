# Neural Physics Subspaces

## 整体架构


我们希望对原论文进行复现，框架则采用XMAKE作为构建配置，Imgui为图形化界面，OpenGL作为渲染器，对于论文中涉及到JAX等自动微分和深度学习库的单元，采用pybind进行打包

```
neural-physics-hybrid/
├── xmake.lua                           # 构建配置
├── src/
|   ├── python/                         # 论文代码
|   │   ├── fem_model.py
|   │   ├── rigid3d_model.py
|   │   ├── integrators.py
|   │   ├── subspace. py
|   │   └── physics_interface.py        # 新增：统一接口
|   ├── 3rdparty/
│   └── VCX/
│       |--Assets/
│       |--Engine/
│       ├--Labs/
│           |--Common/
|           |--Core/                   #这里负责主要逻辑的修改
|               |--python_bridge.h      # pybind11 包装
└── assets/
    ├── shaders/
        ├── mesh.vert
        └── mesh.frag
```