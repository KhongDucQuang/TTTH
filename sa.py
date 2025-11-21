import numpy as np
import time
from costFuncAndUtilities import cost_function, perturb_path, random_path

def sa_pathplanning(env, params):
    # Lấy tham số nhiệt độ
    T0 = params.get("T0", 100.0) # Nhiệt độ thấp thôi
    T = T0
    Tmin = params.get("Tmin", 0.01)
    alpha = params.get("alpha", 0.95)
    max_iter_per_temp = params.get("max_iter_per_temp", 50)
    
    # Khởi tạo đường đi (Sử dụng random_path thông minh từ costFuncAndUtilities)
    current = random_path(env)
    current_cost = cost_function(current, env)
    
    best, best_cost = current.copy(), current_cost
    
    iter_count = 0
    start_time = time.time()
    
    while T > Tmin:
        for _ in range(max_iter_per_temp):
            # === ĐÂY LÀ DÒNG CẦN SỬA ===
            # Trước đây: scale = 150 * (T / T0) + 10  (QUÁ LỚN)
            # Bây giờ: scale giảm dần từ 5.0 xuống 0.5
            # Mục đích: Chỉ rung nhẹ để làm mượt đường cong
            scale = 5.0 * (T / T0) + 0.5
            
            # Tạo đường đi mới bằng cách rung nhẹ
            new_waypoints = perturb_path(current, env, scale)
            new_cost = cost_function(new_waypoints, env)
            
            # Tính chênh lệch năng lượng
            delta = new_cost - current_cost
            
            # Chấp nhận nếu tốt hơn HOẶC chấp nhận xác suất nếu tệ hơn
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current, current_cost = new_waypoints.copy(), new_cost
                
                # Cập nhật Best Solution
                if new_cost < best_cost:
                    best, best_cost = new_waypoints.copy(), new_cost
                    # In ra để theo dõi tiến độ (Optional)
                    # print(f"  >> New Best SA: {best_cost:.2f}")
        
        # Giảm nhiệt độ
        T *= alpha
        iter_count += 1
        
        # In log mỗi 10 vòng nhiệt độ
        if iter_count % 10 == 0:
            print(f"SA Iter {iter_count}: T={T:.2f}, Scale={scale:.2f}, Cost={best_cost:.2f}")

    # Trả về kết quả đầy đủ (Start + Waypoints + Goal)
    full_path = np.vstack([env["start"], best, env["goal"]])
    return full_path, best_cost