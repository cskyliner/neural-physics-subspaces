# Neural_Physics_Subspaces

## 架构

我们希望对原论文进行复现，框架则采用XMAKE作为构建配置，Imgui为图形化界面，OpenGL作为渲染器，对于论文中涉及到JAX等自动微分和深度学习库的单元，采用pybind11进行打包，总体工作流程就是xmake编译，Imgui负责交互和可视化，OpenGL负责渲染，pybind11负责Python与C++的交互；其中交互主要涉及两个方面，一是几何数据的传递，二是物理模拟和训练的调用。对于训练部分则完全使用原代码的Python实现，只是将结果进行加载。

```
neural-physics-hybrid/
├── xmake.lua                           # 构建配置
├── src/
|   ├── python/                         # 物理模拟和训练代码
|   │   ├── fem_model.py
|   │   ├── rigid3d_model.py
|   │   ├── integrators.py
|   │   ├── subspace.py
|   │   |── layers.py
|   │   |── main_learn_subspace.py      # 训练脚本
|   │   |── main_run_system.py          # 运行脚本
|   │   |── minimize.py                 # 优化器实现
|   │   |── physics_interface.py        # 统一接口
|   │   |── system_templates.py         # 系统模板
|   │   |── system_utils.py             # 系统工具函数
|   │   └── utils.py                    # 工具函数
|   ├── 3rdparty/
│   └── VCX/
│       |--Assets/
│       |--Engine/
|           |--GL/                          # 基本渲染
|           |--Python/                      # Python 交互模块
|               |--PythonInterpreter.h      # 全局 Python 解释器单例
|               |--FEMPhysicsBridge.h       # FEM 物理接口包装
|               |--Rigid3DPhysicsBridge.h   # 刚体物理接口包装
|               |--PythonBridge.cpp         # pybind11 包装实现
|               |--FEMPhysicsBridge.cpp     # FEM 物理接口实现
|               |--Rigid3DPhysicsBridge.cpp # 刚体物理接口实现
│       ├--Labs/
│           |--Common/
|           |--Core/                        # 这里负责UI和可视化
|               |--App.cpp                  # 主应用程序
|               |--App.h
|               |--CaseFEM.cpp              # FEM 系统全参数设置模拟实现
|               |--CaseFEM.h
|               |--CaseFEMSubspaces.cpp    # FEM 子空间模拟实现
|               |--CaseFEMSubspaces.h
|               |--CaseRigid3D.cpp         # 刚体系统全参数设置模拟实现
|               |--CaseRigid3D.h
|               |--CaseRigid3DSubspaces.cpp # 刚体子空间模拟实现
|               |--CaseRigid3DSubspaces.h
|               |--ModelLoader.cpp         # 模型加载器实现（便于读取output中的模型数据）
|               |--ModelLoader.h
|               └──main.cpp                 # 程序入口
|── README.md
|── assets/
|   ├── shaders/                            # OpenGL 着色器
|       ├── flat_color.vert                 # 平面着色器顶点程序
|       |── flat_color.frag                 # 平面着色器片段程序
|       |── lit_flat_color.vert             # 光照平面着色器顶点程序
|       └── lit_flat_color.frag             # 光照平面着色器片段程序
|--data/                                    # 物理模拟几何结构数据
|--output/                                  # 记录data-free训练参数和一些相关配置
```

## 可视化