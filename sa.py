import numpy as np
import time
# IMPORT MỚI: Lấy các hàm xịn từ costFuncAndUtilities
from costFuncAndUtilities import cost_function, perturb_path, random_path

def sa_pathplanning(env, params):
    T0 = params.get("T0", 100.0)
    T = T0
    Tmin = params.get("Tmin", 0.01)
    alpha = params.get("alpha", 0.95)
    max_iter_per_temp = params.get("max_iter_per_temp", 50)
    
    # random_path bây giờ đã dùng BFS để tạo đường đi hợp lệ ban đầu
    current = random_path(env)
    current_cost = cost_function(current, env)
    
    best, best_cost = current.copy(), current_cost
    
    iter_count = 0
    
    while T > Tmin:
        for _ in range(max_iter_per_temp):
            # Scale rung lắc (đã được tinh chỉnh trong costFuncAndUtilities)
            # Nhưng ta vẫn truyền scale giảm dần theo nhiệt độ để hội tụ tốt hơn
            current_scale = 1.0 * (T / T0) 
            
            # perturb_path mới xử lý việc không văng ra khỏi map
            new_waypoints = perturb_path(current, env, scale=current_scale)
            new_cost = cost_function(new_waypoints, env)
            
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current, current_cost = new_waypoints.copy(), new_cost
                
                if new_cost < best_cost:
                    best, best_cost = new_waypoints.copy(), new_cost
        
        T *= alpha
        iter_count += 1
        
        # Log nhẹ
        if iter_count % 10 == 0:
            print(f"   SA Iter {iter_count}: Cost={best_cost:.2f}, T={T:.2f}")

    # Trả về full path bao gồm Start và Goal để vẽ
    full_path = np.vstack([env["start"], best, env["goal"]])
    return full_path, best_cost