import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from genWarehouse import generate_warehouse_scenario, plot_warehouse_map
from costFuncAndUtilities import collision_cost, cost_function, smoothness_cost, path_length, perturb_path, random_path, interpolate_path

# ===========================================================
# Hybrid PSO-SA
# ===========================================================
def hybrid_pso_sa(env, params):
    n_particles = params.get("n_particles", 40)
    max_iter = params.get("max_iter", 120)
    w_start, w_end = 0.9, 0.3
    c1, c2 = 2.05, 2.05
    v_max = params.get("v_max", 25)
    
    T0 = params.get("T0_hybrid", 200.0)
    alpha = params.get("alpha_hybrid", 0.96)
    T = T0
    
    positions = []
    for _ in range(n_particles):
        for attempt in range(5):
            p = random_path(env)
            if cost_function(p, env) < 10000:
                positions.append(p)
                break
        else:
            positions.append(random_path(env))
    
    velocities = [np.zeros_like(p) for p in positions]
    
    pbest = [p.copy() for p in positions]
    pbest_cost = np.array([cost_function(p, env) for p in positions])
    
    gbest_idx = np.argmin(pbest_cost)
    gbest = pbest[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]
    
    no_improve_count = 0
    
    for t in range(max_iter):
        w = w_start - (w_start - w_end) * (t / max_iter)
        prev_gbest = gbest_cost
        
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (pbest[i] - positions[i]) + 
                           c2 * r2 * (gbest - positions[i]))
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] += velocities[i]
            positions[i][1:-1, 0] = np.clip(positions[i][1:-1, 0], 0, env["width"])
            positions[i][1:-1, 1] = np.clip(positions[i][1:-1, 1], 0, env["height"])
            
            cost = cost_function(positions[i], env)
            
            delta = cost - pbest_cost[i]
            accept_prob = np.exp(-delta / max(T, 1e-10)) if delta > 0 else 1.0
            
            if np.random.rand() < accept_prob:
                pbest[i] = positions[i].copy()
                pbest_cost[i] = cost
                
                if cost < gbest_cost:
                    gbest = positions[i].copy()
                    gbest_cost = cost
        
        T *= alpha
        
        if abs(prev_gbest - gbest_cost) < 1e-3:
            no_improve_count += 1
        else:
            no_improve_count = 0
            
        if no_improve_count > 15:
            print(f"Hybrid early stopping at iteration {t}")
            break
        
        if t % 10 == 0:
            print(f"Hybrid Iter {t:3d}: T={T:.2f}, best_cost={gbest_cost:.2f}")
    
    return gbest, gbest_cost