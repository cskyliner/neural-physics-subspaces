# Neural_Physics_Subspaces

## 架构

我们希望对原论文进行复现，框架则采用**XMAKE**作为构建配置，**Imgui**为图形化界面，**OpenGL**作为渲染器，对于论文中涉及到**JAX**等**自动微分和深度学习库**的单元，采用**pybind11**进行打包。总体工作流程就是xmake编译，Imgui负责交互和可视化，OpenGL负责渲染，pybind11负责Python与C++的交互；其中交互主要涉及两个方面，一是几何数据的传递，二是物理模拟和训练的调用。对于训练部分则完全使用原代码的Python实现，只是将输出结果进行加载和可视化。

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
│       |--Assets/                          # 资源文件，负责图标和字体的设置
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

## 环境配置与运行方式

**注意**：

强烈建议使用**conda**进行环境管理，以避免依赖冲突问题或者python路径检测失败问题

同时确保安装JAX库，可以参考官方安装：<https://jax.readthedocs.io/en/latest/>

CPU-only (Linux/macOS/Windows)

```bash
pip install -U jax
```

GPU (NVIDIA, CUDA 13)

```bash
pip install -U "jax[cuda13]"
```

### 可视化界面运行

+ 安装XMAKE，参考官网：<https://xmake.io/#/zh-cn/guide/installation>
+ 克隆本仓库到本地
+ 进入项目根目录，使用`conda`创建Python虚拟环境，根据给出的`environment.yml`安装依赖包：

```bash
conda env create -f environment.yml
conda activate subspace_env_clean
```

+ 使用XMAKE编译项目：（因为项目编译内容较多，推荐使用单线程编译以避免并发问题）

```bash
xmake build -j1
```

+ 运行程序：

```bash
xmake run NeuralPhysicsSubspaces
```

+ 在程序界面中选择不同的物理模拟案例进行交互和可视化。

### 训练

参考[原始代码仓库](https://github.com/nmwsharp/neural-physics-subspaces/tree/main)，
使用如下命令进行训练：

```bash
python python/main_learn_subspace.py --system_name [system_name] --problem_name [problem_name] --subspace_dim=8 --weight_expand=1.0 --sigma_scale=1.0 --output_dir output/
```

其中`[system_name]`和`[problem_name]`可以参考`system_templates.py`中的定义。而`subspace_dim`，`weight_expand`和`sigma_scale`均为超参数，可以根据需要进行调整。具体调整方式可以查看[论文](https://arxiv.org/abs/2305.03846)。
