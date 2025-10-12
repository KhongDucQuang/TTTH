import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from genWarehouse import generate_warehouse_scenario, plot_warehouse_map
# ===========================================================
# Cost function & Utilities cho obstacles hình chữ nhật
# ===========================================================

def interpolate_path(path, num_points=8):
    """Nội suy đường đi để kiểm tra collision chi tiết hơn"""
    interpolated_path = []
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        points = np.linspace(p1, p2, num_points)
        interpolated_path.extend(points[:-1])
    interpolated_path.append(path[-1])
    return np.array(interpolated_path)

def path_length(path):
    """Tính độ dài đường đi"""
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def smoothness_cost(path):
    """Tính độ mượt dựa trên góc quẹo"""
    if len(path) < 3:
        return 0
    slopes = np.diff(path, axis=0)
    angles = np.arctan2(slopes[:, 1], slopes[:, 0])
    return np.sum(np.abs(np.diff(np.unwrap(angles))))

def collision_cost(path, env):
    """
    Tính collision cost cho obstacles hình chữ nhật
    """
    detailed_path = interpolate_path(path, num_points=10)
    
    total = 0.0
    margin = env["safety_margin"]
    
    for obs_data in env["obstacles_rect"]:
        x_min, y_min, width, height, _ = obs_data
        x_max = x_min + width
        y_max = y_min + height
        
        for point in detailed_path:
            px, py = point
            
            # Tìm điểm gần nhất trên hình chữ nhật
            closest_x = np.clip(px, x_min, x_max)
            closest_y = np.clip(py, y_min, y_max)
            
            # Tính khoảng cách đến hình chữ nhật
            dist = np.linalg.norm([px - closest_x, py - closest_y])
            
            # Kiểm tra xâm nhập CORE (bên trong hình chữ nhật)
            if dist == 0:
                # Điểm nằm trong hoặc trên viền hình chữ nhật
                total += 50000  # Phạt CỰC NẶNG
            elif dist < margin:
                # Phạt nếu trong vùng safety
                penetration = margin - dist
                total += (penetration**3) * 20
    
    return total

def cost_function(path, env):
    """Hàm chi phí tổng hợp"""
    w1, w2, w3 = env["weights"]
    
    length = path_length(path)
    smooth = smoothness_cost(path)
    collision = collision_cost(path, env)
    
    return w1 * length + w2 * smooth + w3 * collision

def random_path(env):
    """Tạo đường đi ngẫu nhiên hợp lý - TRÁNH vật cản hình chữ nhật"""
    start, goal = env["start"], env["goal"]
    n = env["n_waypoints"]
    
    max_attempts = 100
    for attempt in range(max_attempts):
        direction = goal - start
        mids = []
        valid = True
        
        for i in range(n):
            t = (i + 1) / (n + 1)
            base_point = start + t * direction
            noise = np.random.randn(2) * 100
            point = base_point + noise
            point[0] = np.clip(point[0], 30, env["width"] - 30)
            point[1] = np.clip(point[1], 30, env["height"] - 30)
            
            # Kiểm tra va chạm với obstacles hình chữ nhật
            for obs_data in env["obstacles_rect"]:
                x_min, y_min, width, height, _ = obs_data
                x_max = x_min + width
                y_max = y_min + height
                margin = env["safety_margin"]
                
                # Kiểm tra xem điểm có nằm trong vùng mở rộng không
                if (x_min - margin <= point[0] <= x_max + margin and
                    y_min - margin <= point[1] <= y_max + margin):
                    valid = False
                    break
            
            if not valid:
                break
            mids.append(point)
        
        if valid and len(mids) == n:
            return np.vstack([start, np.array(mids), goal])
    
    # Fallback: đường thẳng
    mids = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        mids.append(start + t * (goal - start))
    return np.vstack([start, np.array(mids), goal])

def perturb_path(path, env, scale=30):
    """Tạo nhiễu cho đường đi"""
    new = path.copy()
    idx = np.random.randint(1, len(path)-1)
    new[idx] += np.random.randn(2) * scale
    new[idx, 0] = np.clip(new[idx, 0], 0, env["width"])
    new[idx, 1] = np.clip(new[idx, 1], 0, env["height"])
    return new