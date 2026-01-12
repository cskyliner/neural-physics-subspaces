"""
统一的 C++ 调用接口
"""
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional

import config
import integrators
import subspace

class PhysicsInterface:
    """为 C++ 提供的统一物理接口"""
    
    def __init__(self, system_name: str, problem_name: str, 
                 subspace_model: Optional[str] = None):
        # 构建系统
        self.system, self.system_def = config.construct_system_from_name(
            system_name, problem_name
        )
        
        # 加载子空间
        self.subspace_model = None
        self.subspace_dim = -1
        self.use_subspace = False
        
        if subspace_model:
            self.load_subspace(subspace_model)
        
        # 初始化状态
        self.reset_state()
        
        # 积分器设置
        self.int_opts = {}
        self.int_state = {}
        integrators.initialize_integrator(
            self.int_opts, self.int_state, "implicit-proximal"
        )
    
    def load_subspace(self, model_path: str):
        """加载子空间模型"""
        import json
        import equinox as eqx
        from layers import create_model
        
        # 加载模型
        with open(model_path + '. json', 'r') as f:
            spec = json.load(f)
        
        model = create_model(spec)
        _, model_static = eqx.partition(model, eqx.is_array)
        model_params = eqx.tree_deserialise_leaves(model_path + ". eqx", model)
        
        # 加载元数据
        info = np.load(model_path + "_info.npy", allow_pickle=True).item()
        
        self.subspace_dim = info['subspace_dim']
        self.subspace_domain_dict = subspace.get_subspace_domain_dict(
            info['subspace_domain_type']
        )
        self.t_schedule_final = info['t_schedule_final']
        
        # 创建应用函数
        def apply_fn(x, cond_params):
            m = eqx.combine(model_params, model_static)
            return m(jnp.concatenate((x, cond_params), axis=-1),
                    t_schedule=self.t_schedule_final)
        
        self.subspace_apply = jax.jit(apply_fn)
        self.use_subspace = True
        
        print(f"Loaded subspace:  {self.subspace_dim}D")
    
    def reset_state(self):
        """重置状态"""
        if self.use_subspace:
            initial_val = self.subspace_domain_dict['initial_val']
            self. q_t = jnp.zeros(self.subspace_dim) + initial_val
        else:
            self.q_t = self.system_def['init_pos']
        
        self.q_tm1 = self.q_t
        self. qdot_t = jnp.zeros_like(self.q_t)
    
    def state_to_system(self, q):
        """将状态映射到系统空间"""
        if self.use_subspace:
            return self.subspace_apply(q, self.system_def['cond_param'])
        return q
    
    def get_visualization_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取可视化数据
        返回:  (vertices, faces) 
        vertices: (N, 3) float64
        faces: (M, 3) int32
        """
        q_full = self.state_to_system(self.q_t)
        vertices, faces = self.system.get_mesh(self.system_def, q_full)
        
        # 确保是 numpy 数组
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int32)
        
        return vertices, faces
    
    def timestep(self, dt: float = 0.05):
        """执行一个时间步"""
        # 更新积分器状态
        self.int_state['q_t'] = self.q_t
        self.int_state['q_tm1'] = self.q_tm1
        self.int_state['qdot_t'] = self.qdot_t
        
        # 执行积分
        subspace_fn = self.state_to_system if self.use_subspace else None
        domain_dict = self.subspace_domain_dict if self.use_subspace else None
        
        self. int_state = integrators.timestep(
            self.system,
            self.system_def,
            self.int_state,
            self.int_opts,
            subspace_fn=subspace_fn,
            subspace_domain_dict=domain_dict
        )
        
        # 更新状态
        self.q_t = self.int_state['q_t']
        self.q_tm1 = self.int_state['q_tm1']
        self.qdot_t = self.int_state['qdot_t']
    
    def get_potential_energy(self) -> float:
        """获取势能"""
        q_full = self.state_to_system(self.q_t)
        E = self.system.potential_energy(self.system_def, q_full)
        return float(E)
    
    def get_state(self) -> np.ndarray:
        """获取当前状态"""
        return np.array(self.q_t, dtype=np.float64)
    
    def set_state(self, q: np.ndarray):
        """设置状态"""
        self. q_t = jnp.array(q)
        if self.use_subspace:
            # 应用域投影
            self.q_t = subspace.apply_domain_projection(
                self.q_t, self.subspace_domain_dict
            )
    
    def stop_velocity(self):
        """停止速度"""
        self.qdot_t = jnp.zeros_like(self. qdot_t)
    
    def get_info(self) -> dict:
        """获取系统信息"""
        return {
            'full_dim': int(self.system_def['init_pos'].shape[0]),
            'subspace_dim': int(self.subspace_dim) if self.use_subspace else -1,
            'use_subspace': bool(self.use_subspace),
            'domain_type': self.subspace_domain_dict. get('domain_name', 'none') if self.use_subspace else 'none',
        }