import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from genWarehouse import generate_warehouse_scenario, plot_warehouse_map
from costFuncAndUtilities import collision_cost, cost_function, smoothness_cost, path_length, perturb_path, random_path, interpolate_path


# ===========================================================
# Simulated Annealing
# ===========================================================
def sa_pathplanning(env, params):
    T0 = params.get("T0", 2000.0)
    T = T0
    Tmin = params.get("Tmin", 0.1)
    alpha = params.get("alpha", 0.96)
    max_iter_per_temp = params.get("max_iter_per_temp", 40)
    
    # Khởi tạo với nhiều lần thử
    best_init_cost = float('inf')
    for _ in range(10):
        candidate = random_path(env)
        candidate_cost = cost_function(candidate, env)
        if candidate_cost < best_init_cost:
            current = candidate
            current_cost = candidate_cost
            best_init_cost = candidate_cost
    
    best, best_cost = current.copy(), current_cost
    no_improve_count = 0
    
    iter_count = 0
    while T > Tmin:
        improved_this_temp = False
        for _ in range(max_iter_per_temp):
            scale = 60 * (T / T0) + 10
            new = perturb_path(current, env, scale)
            new_cost = cost_function(new, env)
            
            delta = new_cost - current_cost
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current, current_cost = new.copy(), new_cost
                if new_cost < best_cost:
                    best, best_cost = new.copy(), new_cost
                    improved_this_temp = True
                    no_improve_count = 0
        
        if not improved_this_temp:
            no_improve_count += 1
        
        if no_improve_count > 20:
            print(f"SA early stopping at T={T:.2f}")
            break
            
        T *= alpha
        iter_count += 1
        if iter_count % 10 == 0:
            print(f"SA Iter {iter_count}: T={T:.2f}, best_cost={best_cost:.2f}")
    
    return best, best_cost