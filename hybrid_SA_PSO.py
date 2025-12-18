import numpy as np
# Cần import thêm perturb_path để dùng cho việc rung lắc gBest
from costFuncAndUtilities import cost_function, random_path, perturb_path

def hybrid_pso_sa(env, params):
    n_particles = params.get("n_particles", 40)
    max_iter = params.get("max_iter", 120)
    w_start, w_end = 0.9, 0.4
    c1, c2 = 2.0, 2.0
    v_max = params.get("v_max", 2.0)
    
    T0 = params.get("T0_hybrid", 100.0)
    alpha = params.get("alpha_hybrid", 0.95)
    T = T0
    
    # Số lần thử SA trên gBest mỗi vòng lặp (Local Search iterations)
    L_sa = 5 
    
    positions = [random_path(env) for _ in range(n_particles)]
    velocities = [np.zeros_like(p) for p in positions]
    
    pbest = [p.copy() for p in positions]
    pbest_cost = np.array([cost_function(p, env) for p in positions])
    
    gbest_idx = np.argmin(pbest_cost)
    gbest = pbest[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]
    
    for t in range(max_iter):
        w = w_start - (w_start - w_end) * (t / max_iter)
        
        # --- BƯỚC 1: PSO STANDARD UPDATE (Di chuyển bầy đàn) ---
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest[i] - positions[i]) + 
                             c2 * r2 * (gbest - positions[i]))
            
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] += velocities[i]
            
            positions[i][:, 0] = np.clip(positions[i][:, 0], 0, env["width"])
            positions[i][:, 1] = np.clip(positions[i][:, 1], 0, env["height"])
            
            cost = cost_function(positions[i], env)
            
            # [SỬA LẠI] Quay về logic chuẩn của PSO: Chỉ cập nhật nếu tốt hơn
            # Để đảm bảo tính hội tụ nhanh (Exploitation)
            if cost < pbest_cost[i]:
                pbest[i] = positions[i].copy()
                pbest_cost[i] = cost
                
                if cost < gbest_cost:
                    gbest = positions[i].copy()
                    gbest_cost = cost
        
        # --- BƯỚC 2: SA LOCAL SEARCH ON gBEST (Tinh chỉnh gBest) ---
        # Đây là phần "Hồn" của bài báo gốc.
        # Sau khi cả bầy tìm được gBest, ta dùng SA để "đào sâu" xung quanh gBest đó.
        
        # Scale rung lắc giảm dần theo nhiệt độ (càng về sau càng tinh chỉnh nhỏ)
        current_scale = 1.0 * (T / T0) + 0.1
        
        for _ in range(L_sa):
            # Tạo ứng viên mới từ gBest bằng cách rung lắc nhẹ
            candidate_pos = perturb_path(gbest, env, scale=current_scale)
            candidate_cost = cost_function(candidate_pos, env)
            
            delta = candidate_cost - gbest_cost
            
            # Logic SA:
            # 1. Nếu tốt hơn: Chắc chắn chọn -> Cập nhật gBest
            # 2. Nếu tệ hơn: Chấp nhận xác suất để THOÁT CỰC TIỂU ĐỊA PHƯƠNG
            if delta < 0:
                gbest = candidate_pos.copy()
                gbest_cost = candidate_cost
            else:
                r = np.random.rand()
                if r < np.exp(-delta / max(T, 1e-10)):
                    # LƯU Ý QUAN TRỌNG:
                    # Khi chấp nhận kết quả tệ hơn, ta cập nhật gBest TẠM THỜI
                    # để ở vòng sau, các hạt khác sẽ bị kéo về hướng mới này (thoát bẫy).
                    # Tuy nhiên, để an toàn, ta thường giữ một bản backup "Best So Far"
                    # nhưng trong code đơn giản này, ta cho phép gBest bị trôi đi
                    # để tăng khả năng Exploration.
                    gbest = candidate_pos.copy()
                    gbest_cost = candidate_cost
        
        # Giảm nhiệt độ
        T *= alpha
        
        if t % 20 == 0:
            print(f"   Hybrid Iter {t:3d}: T={T:.2f}, Cost={gbest_cost:.2f}")

    full_path = np.vstack([env["start"], gbest, env["goal"]])
    return full_path, gbest_cost