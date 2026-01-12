#!/usr/bin/env python3
"""
测试FEM系统是否可以从C++正确调用
"""

import sys
sys.path.append('./python')

import numpy as np
import jax.numpy as jnp
from fem_model import FEMSystem
import integrators

def test_fem_system(problem_name="bistable"):
    print(f"[Test] Loading FEM system: {problem_name}")
    
    # 构建系统
    system, system_def = FEMSystem.construct(problem_name)
    
    print(f"[Test] System loaded successfully")
    print(f"  - Full dimension: {system_def['init_pos'].size}")
    print(f"  - Position dimension: {system.pos_dim}")
    print(f"  - Problem name: {system.problem_name}")
    
    # 初始化积分器
    int_opts = {}
    int_state = {}
    integrators.initialize_integrator(int_opts, int_state, "implicit_euler")
    
    # 设置初始状态
    int_state['q_t'] = system_def['init_pos']
    int_state['q_tm1'] = system_def['init_pos']
    int_state['qdot_t'] = jnp.zeros_like(system_def['init_pos'])
    
    print(f"[Test] Integrator initialized")
    
    # 执行几个时间步
    print(f"[Test] Running simulation...")
    for i in range(5):
        int_state = integrators.timestep(
            system, system_def, int_state, int_opts,
            subspace_fn=None, subspace_domain_dict=None
        )
        
        # 获取当前位置
        pos = system.get_full_position(system, system_def, int_state['q_t'])
        
        # 计算能量
        energy = system.potential_energy(system_def, int_state['q_t'])
        
        print(f"  Step {i+1}: Energy = {energy:.6f}")
    
    # 获取可视化数据
    print(f"[Test] Getting visualization data...")
    pos = system.get_full_position(system, system_def, int_state['q_t'])
    mesh = system.mesh
    
    print(f"  - Vertices shape: {np.array(pos).shape}")
    print(f"  - Vertices count: {len(pos)}")
    
    if system.pos_dim == 2:
        faces = mesh['E']
        print(f"  - Faces shape: {np.array(faces).shape}")
    else:
        if 'boundary_triangles' in mesh:
            faces = mesh['boundary_triangles']
            print(f"  - Boundary faces shape: {np.array(faces).shape}")
        else:
            print(f"  - Warning: No boundary triangles found")
    
    # 测试外力设置
    if 'pull_X' in system_def['external_forces']:
        print(f"[Test] Testing external forces...")
        system_def['external_forces']['pull_X'] = jnp.array(1.0)
        system_def['external_forces']['pull_strength'] = 0.05
        print(f"  - External force set successfully")
    
    print(f"[Test] All tests passed!")
    return True

if __name__ == '__main__':
    problems = ["bistable", "load3d", "heterobeam"]
    
    for problem in problems:
        try:
            print(f"\n{'='*60}")
            test_fem_system(problem)
        except Exception as e:
            print(f"[Error] Failed to test {problem}: {e}")
            import traceback
            traceback.print_exc()
