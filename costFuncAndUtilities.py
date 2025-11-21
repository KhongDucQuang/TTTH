import numpy as np
from scipy.interpolate import splprep, splev
import heapq

# =========================================================
# 1. HÀM TẠO B-SPLINE (ROBUST - KHÔNG BAO GIỜ LỖI)
# =========================================================

def generate_bspline_path(full_path, n_points=100, degree=3):
    """
    Tạo đường cong B-Spline từ các điểm điều khiển (waypoints).
    Đã xử lý triệt để lỗi 'mất đường cong'.
    """
    # Nếu quá ít điểm, trả về đường thẳng nội suy
    if len(full_path) < 2:
        return interpolate_path(full_path, n_points)

    # BƯỚC 1: Lọc bỏ các điểm trùng nhau (nguyên nhân chính gây lỗi)
    valid_path = [full_path[0]]
    for i in range(1, len(full_path)):
        dist = np.linalg.norm(full_path[i] - valid_path[-1])
        if dist > 1.0: # Chỉ giữ lại điểm cách nhau > 1cm
            valid_path.append(full_path[i])
    valid_path = np.array(valid_path)

    # BƯỚC 2: Nếu sau khi lọc còn quá ít điểm, hoặc bậc spline quá cao
    # Giảm bậc spline xuống
    curr_degree = degree
    if len(valid_path) <= degree:
        curr_degree = len(valid_path) - 1
    
    if curr_degree < 1: # Không thể tạo spline
        return interpolate_path(full_path, n_points)

    # BƯỚC 3: Thêm nhiễu cực nhỏ (Jitter) để tránh lỗi toán học khi thẳng hàng
    # Scipy splprep sẽ crash nếu các điểm thẳng hàng tuyệt đối
    jitter = np.random.normal(0, 1e-4, valid_path.shape)
    valid_path = valid_path + jitter

    try:
        # Tạo B-Spline
        tck, u = splprep([valid_path[:, 0], valid_path[:, 1]], s=0, k=curr_degree)
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = splev(u_new, tck)
        return np.c_[x_new, y_new]
    except Exception as e:
        # Fallback cuối cùng: Nối thẳng
        print(f"[Warning] Spline failed: {e}")
        return interpolate_path(full_path, n_points)

def interpolate_path(path, num_points=10):
    """Nối thẳng các điểm (Linear Interpolation)"""
    if len(path) < 2: return path
    dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_dist = np.sum(dists)
    if total_dist == 0: return np.array([path[0]] * num_points)
    
    cum_dist = np.r_[0, np.cumsum(dists)]
    sample_dists = np.linspace(0, total_dist, num_points)
    
    x = np.interp(sample_dists, cum_dist, path[:, 0])
    y = np.interp(sample_dists, cum_dist, path[:, 1])
    return np.c_[x, y]

# =========================================================
# 2. HÀM KHỞI TẠO (INITIALIZATION) - THEO BÀI BÁO
# =========================================================
# Bài báo yêu cầu quần thể ban đầu phải nằm trong vùng khả thi.
# Vì map quá khó (cửa hẹp), ta dùng thuật toán loang (Flood Fill/BFS) 
# trên lưới mịn để tìm ra "Topology" (khung xương) hợp lệ ban đầu.

class BinaryGrid:
    def __init__(self, env, resolution=5): # Lưới 5cm cực mịn
        self.res = resolution
        self.w = int(env["width"] / resolution) + 1
        self.h = int(env["height"] / resolution) + 1
        self.grid = np.zeros((self.w, self.h), dtype=bool)
        
        # Đánh dấu tường (Nở rộng tường ra một chút để an toàn)
        margin = 10.0 # 5cm margin an toàn cho khởi tạo
        for wall in env["walls_rect"]:
            x, y, w, h = wall
            ix_min = max(0, int((x - margin) / resolution))
            iy_min = max(0, int((y - margin) / resolution))
            ix_max = min(self.w, int((x + w + margin) / resolution))
            iy_max = min(self.h, int((y + h + margin) / resolution))
            self.grid[ix_min:ix_max, iy_min:iy_max] = True # True là tường

    def is_wall(self, ix, iy):
        if ix < 0 or ix >= self.w or iy < 0 or iy >= self.h: return True
        return self.grid[ix, iy]

    def get_valid_start_goal(self, start, goal):
        # Dời start/goal nếu lỡ nằm trong tường
        s_node = self.find_nearest_free(start)
        g_node = self.find_nearest_free(goal)
        return s_node, g_node

    def find_nearest_free(self, pos):
        ix, iy = int(pos[0]/self.res), int(pos[1]/self.res)
        if not self.is_wall(ix, iy): return (ix, iy)
        # Tìm xoắn ốc
        for r in range(1, 50):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if self.is_wall(ix+dx, iy+dy) == False:
                        return (ix+dx, iy+dy)
        return (ix, iy)

def bfs_init_path(env):
    """
    Tìm một đường đi hợp lệ trên lưới để làm khung khởi tạo.
    Dùng BFS đơn giản (nhanh hơn A* và đủ tốt cho init).
    """
    bg = BinaryGrid(env, resolution=5)
    start_idx, goal_idx = bg.get_valid_start_goal(env["start"], env["goal"])
    
    queue = [(start_idx, [start_idx])]
    visited = set([start_idx])
    
    # Hướng đi: 8 hướng
    moves = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    path_indices = []
    
    # Giới hạn số bước để không treo máy
    steps = 0
    found = False
    
    while queue and steps < 50000:
        (cx, cy), path = queue.pop(0) # BFS lấy đầu hàng đợi
        steps += 1
        
        # Check Goal (gần đúng)
        if abs(cx - goal_idx[0]) < 2 and abs(cy - goal_idx[1]) < 2:
            path_indices = path
            found = True
            break
            
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited and not bg.is_wall(nx, ny):
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
                
    if not found:
        # Fallback: Đường thẳng (chấp nhận rủi ro nếu map quá khó)
        return np.linspace(env["start"], env["goal"], env["n_waypoints"] + 2)

    # Convert Grid Index -> World Coordinates
    raw_path = np.array(path_indices) * bg.res
    
    # Lấy mẫu lại (Resample) để có đúng n_waypoints điểm
    # Kỹ thuật: Tính tổng độ dài -> chia đều khoảng cách
    dists = np.linalg.norm(np.diff(raw_path, axis=0), axis=1)
    cum_dist = np.r_[0, np.cumsum(dists)]
    target_dists = np.linspace(0, cum_dist[-1], env["n_waypoints"] + 2)
    
    final_path = np.zeros((env["n_waypoints"] + 2, 2))
    final_path[:, 0] = np.interp(target_dists, cum_dist, raw_path[:, 0])
    final_path[:, 1] = np.interp(target_dists, cum_dist, raw_path[:, 1])
    
    # Bỏ start và goal, chỉ trả về waypoints ở giữa
    return final_path[1:-1]

def random_path(env):
    """
    Hàm khởi tạo được gọi bởi PSO/SA.
    Tạo ra một đường đi hợp lệ + Nhiễu ngẫu nhiên.
    """
    # 1. Lấy khung đường đi hợp lệ từ BFS
    base_waypoints = bfs_init_path(env)
    
    # 2. Thêm nhiễu (Randomness) để tạo quần thể đa dạng cho PSO
    # Nhiễu khoảng 10-20cm để các hạt khác nhau
    noise = np.random.normal(0, 5.0, base_waypoints.shape)
    waypoints = base_waypoints + noise
    
    # Clip để không văng ra ngoài map
    waypoints[:, 0] = np.clip(waypoints[:, 0], 0, env['width'])
    waypoints[:, 1] = np.clip(waypoints[:, 1], 0, env['height'])
    
    return waypoints

# =========================================================
# 3. COST FUNCTION - THEO CÔNG THỨC BÀI BÁO
# =========================================================

def check_collision_rect_fast(pts, walls, margin):
    """Kiểm tra va chạm nhanh cho mảng điểm."""
    for x, y, w, h in walls:
        # Vector hóa check: Tìm các điểm nằm trong HCN mở rộng
        mask = (pts[:,0] >= x - margin) & (pts[:,0] <= x + w + margin) & \
               (pts[:,1] >= y - margin) & (pts[:,1] <= y + h + margin)
        if np.any(mask): return True
    return False

def cost_function(waypoints, env):
    # Ghép điểm Start/Goal vào
    full_path = np.vstack([env["start"], waypoints, env["goal"]])
    
    # QUAN TRỌNG: Tạo đường cong B-Spline để đánh giá
    # Bài báo tối ưu trên đường cong, không phải đường gấp khúc
    bspline = generate_bspline_path(full_path, n_points=100, degree=3)
    
    # Trọng số (Theo bài báo thường ưu tiên An toàn > Mượt > Ngắn)
    weights = env.get("weights", [1.0, 10.0, 100000.0, 0.0])
    w_len, w_smooth, w_coll, w_dist = weights
    
    # 1. Cost Va chạm (Collision) - Penalty Method
    # Margin cứng = 15cm
    c_coll = 0
    margin = env.get("safety_margin_hard", 15.0)
    walls = env["walls_rect"]
    
    # Check từng điểm trên B-spline
    for p in bspline:
        # Check biên map
        if p[0] < margin or p[0] > env["width"]-margin or p[1] < margin or p[1] > env["height"]-margin:
             c_coll += 1
             continue
        # Check tường
        hit = False
        for wx, wy, ww, wh in walls:
            if (p[0] >= wx - margin and p[0] <= wx + ww + margin and
                p[1] >= wy - margin and p[1] <= wy + wh + margin):
                c_coll += 1
                hit = True
                break
    
    # Nếu va chạm -> Phạt cực nặng (Exponential Penalty)
    if c_coll > 0:
        return 1e6 * c_coll # 1 triệu điểm phạt mỗi lần chạm
        
    # 2. Cost Độ dài (Length)
    c_len = np.sum(np.linalg.norm(np.diff(bspline, axis=0), axis=1))
    
    # 3. Cost Độ mượt (Smoothness) - Tổng thay đổi góc
    slopes = np.diff(bspline, axis=0)
    norms = np.linalg.norm(slopes, axis=1)
    # Tránh chia cho 0
    valid = norms > 1e-6
    if np.sum(valid) >= 2:
        slopes = slopes[valid] / norms[valid, None]
        angles = np.arctan2(slopes[:, 1], slopes[:, 0])
        c_smooth = np.sum(np.abs(np.diff(np.unwrap(angles))))
    else:
        c_smooth = 0
        
    return w_len * c_len + w_smooth * c_smooth

def perturb_path(waypoints, env, scale=1.0):
    """Hàm rung lắc cho SA"""
    new_wp = waypoints.copy()
    idx = np.random.randint(0, len(new_wp))
    new_wp[idx] += np.random.normal(0, scale, 2) # Gaussian noise hay hơn Uniform
    
    new_wp[:, 0] = np.clip(new_wp[:, 0], 0, env['width'])
    new_wp[:, 1] = np.clip(new_wp[:, 1], 0, env['height'])
    return new_wp