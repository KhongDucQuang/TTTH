import numpy as np
from costFuncAndUtilities import cost_function, random_path

def hybrid_pso_sa(env, params):
    n_particles = params.get("n_particles", 40)
    max_iter = params.get("max_iter", 120)
    w_start, w_end = 0.9, 0.4
    c1, c2 = 2.0, 2.0
    v_max = params.get("v_max", 25)
    
    T0 = params.get("T0_hybrid", 200.0)
    alpha = params.get("alpha_hybrid", 0.96)
    T = T0
    
    # Các biến chỉ chứa waypoints
    positions = [random_path(env) for _ in range(n_particles)]
    velocities = [np.zeros_like(p) for p in positions]
    
    pbest = [p.copy() for p in positions]
    pbest_cost = np.array([cost_function(p, env) for p in positions])
    
    gbest_idx = np.argmin(pbest_cost)
    gbest = pbest[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]
    
    for t in range(max_iter):
        w = w_start - (w_start - w_end) * (t / max_iter)
        
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            
            # Cập nhật vận tốc và vị trí cho các waypoints
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest[i] - positions[i]) + 
                             c2 * r2 * (gbest - positions[i]))
            
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] += velocities[i]
            
            positions[i][:, 0] = np.clip(positions[i][:, 0], 0, env["width"])
            positions[i][:, 1] = np.clip(positions[i][:, 1], 0, env["height"])
            
            cost = cost_function(positions[i], env)
            
            # Cơ chế chấp nhận của SA được áp dụng cho pbest
            delta = cost - pbest_cost[i]
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                pbest[i] = positions[i].copy()
                pbest_cost[i] = cost
                if cost < gbest_cost:
                    gbest = positions[i].copy()
                    gbest_cost = cost
        
        T *= alpha
        
        if t % 10 == 0:
            print(f"Hybrid Iter {t:3d}: T={T:.2f}, best_cost={gbest_cost:.2f}")

    # Trả về đường đi hoàn chỉnh để vẽ
    full_path = np.vstack([env["start"], gbest, env["goal"]])
    return full_path, gbest_cost