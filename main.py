import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# ===========================================================
# 1️⃣ Sinh môi trường warehouse theo scenario mới
# ===========================================================
def generate_warehouse_scenario():
    """
    Tạo bản đồ warehouse theo hình mẫu với các khu vực A-G
    """
    env = {}
    env["width"], env["height"] = 900, 500
    
    # Định nghĩa các khu vực chứa hàng (obstacles)
    # Format: [x_min, y_min, width, height, name]
    env["obstacles_rect"] = [
        [500, 320, 170, 130, 'A'],  # Area A (phía trên giữa)
        [500, 130, 250, 120, 'B'],  # Area B (giữa)
        [500, 10, 250, 100, 'C'],   # Area C (dưới)
        [800, 310, 100, 190, 'D'],  # Area D (phía trên bên phải)
        [800, 190, 100, 110, 'E'],  # Area E (giữa bên phải)
        [750, 10, 150, 70, 'F'],    # Area F (dưới bên phải)
    ]
    
    # Area G (khu trống) - chỉ để vẽ, không phải obstacle
    env["area_g"] = [50, 50, 400, 400, 'G']
    
    # Điểm start và goal
    env["start"] = np.array([50, 450])   # Góc trên bên trái
    env["goal"] = np.array([850, 50])    # Góc dưới bên phải
    
    env["n_waypoints"] = 6
    env["safety_margin"] = 15
    env["weights"] = np.array([1.0, 0.5, 100.0])  # length, smoothness, collision
    
    return env

def plot_warehouse_map(env, show_grid=True):
    """
    Vẽ bản đồ warehouse scenario
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, env["width"])
    ax.set_ylim(0, env["height"])
    ax.set_xlabel('X (cm)', fontsize=11)
    ax.set_ylabel('Y (cm)', fontsize=11)
    ax.set_title('Warehouse Scenario', fontsize=14, fontweight='bold')
    
    # Vẽ Area G (khu trống) - chỉ đường viền
    if 'area_g' in env:
        x, y, w, h, name = env["area_g"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, f'Area {name}', 
               ha='center', va='center', fontsize=13, color='blue')
    
    # Vẽ các khu vực chứa hàng (obstacles)
    for obs_data in env["obstacles_rect"]:
        x, y, w, h, name = obs_data
        
        # Vẽ hình chữ nhật
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor='black', facecolor='lightgray', alpha=0.5)
        ax.add_patch(rect)
        
        # Vẽ đường viền safety margin (nét đứt)
        margin = env["safety_margin"]
        safety_rect = patches.Rectangle((x - margin, y - margin), 
                                        w + 2*margin, h + 2*margin, 
                                        linewidth=1, edgecolor='orange', 
                                        facecolor='yellow', linestyle='--', alpha=0.2)
        ax.add_patch(safety_rect)
        
        # Thêm tên khu vực
        ax.text(x + w/2, y + h/2, f'Area {name}', 
               ha='center', va='center', fontsize=13, color='blue', fontweight='bold')
    
    # Vẽ lưới các điểm (dots) giống hình mẫu
    if show_grid:
        grid_x = np.arange(50, env["width"], 50)
        grid_y = np.arange(50, env["height"], 50)
        for gx in grid_x:
            for gy in grid_y:
                ax.plot(gx, gy, 'k.', markersize=2)
    
    # Vẽ start và goal
    ax.plot(*env["start"], 'go', markersize=14, label='Start', zorder=10)
    ax.plot(*env["goal"], 'ro', markersize=14, label='Goal', zorder=10)
    
    ax.legend(loc='upper left', fontsize=11)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.grid(False)
    plt.show()

# ===========================================================
# 2️⃣ Cost function & Utilities cho obstacles hình chữ nhật
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

# ===========================================================
# 3️⃣ Simulated Annealing
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

# ===========================================================
# 4️⃣ Particle Swarm Optimization
# ===========================================================
def pso_pathplanning(env, params):
    n_particles = params.get("n_particles", 40)
    max_iter = params.get("max_iter", 120)
    w_start, w_end = 0.9, 0.3
    c1, c2 = 2.05, 2.05
    v_max = params.get("v_max", 25)
    
    # Khởi tạo
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
            
            if cost < pbest_cost[i]:
                pbest[i] = positions[i].copy()
                pbest_cost[i] = cost
                
                if cost < gbest_cost:
                    gbest = positions[i].copy()
                    gbest_cost = cost
        
        if abs(prev_gbest - gbest_cost) < 1e-3:
            no_improve_count += 1
        else:
            no_improve_count = 0
            
        if no_improve_count > 15:
            print(f"PSO early stopping at iteration {t}")
            break
        
        if t % 10 == 0:
            print(f"PSO Iter {t:3d}: best_cost={gbest_cost:.2f}")
    
    return gbest, gbest_cost

# ===========================================================
# 5️⃣ Hybrid PSO-SA
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

# ===========================================================
# 6️⃣ Hiển thị kết quả
# ===========================================================
def plot_result(env, path, title, cost):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, env["width"])
    ax.set_ylim(0, env["height"])
    ax.set_xlabel('X (cm)', fontsize=11)
    ax.set_ylabel('Y (cm)', fontsize=11)
    
    # Vẽ Area G
    if 'area_g' in env:
        x, y, w, h, name = env["area_g"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, f'Area {name}', 
               ha='center', va='center', fontsize=11, color='blue', alpha=0.5)
    
    # Vẽ obstacles
    for obs_data in env["obstacles_rect"]:
        x, y, w, h, name = obs_data
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor='black', facecolor='lightgray', alpha=0.5)
        ax.add_patch(rect)
        
        margin = env["safety_margin"]
        safety_rect = patches.Rectangle((x - margin, y - margin), 
                                        w + 2*margin, h + 2*margin, 
                                        linewidth=1, edgecolor='orange', 
                                        facecolor='yellow', linestyle='--', alpha=0.15)
        ax.add_patch(safety_rect)
        
        ax.text(x + w/2, y + h/2, f'Area {name}', 
               ha='center', va='center', fontsize=11, color='blue', alpha=0.7)
    
    # Vẽ đường đi
    ax.plot(path[:,0], path[:,1], 'b-', linewidth=2.5, alpha=0.8, label='Path', zorder=5)
    ax.plot(path[:,0], path[:,1], 'co', markersize=7, zorder=6)
    
    # Vẽ start và goal
    ax.plot(*env["start"], 'go', markersize=14, label='Start', zorder=10)
    ax.plot(*env["goal"], 'mo', markersize=14, label='Goal', zorder=10)
    
    full_title = f"{title}\nCost: {cost:.2f}"
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# ===========================================================
# 7️⃣ Main test
# ===========================================================
if __name__ == "__main__":
    print("="*70)
    print("WAREHOUSE PATH PLANNING - PSO, SA, HYBRID COMPARISON")
    print("="*70)
    
    # Tạo môi trường
    env = generate_warehouse_scenario()
    
    print("\n📍 Environment Info:")
    print(f"   Size: {env['width']} x {env['height']} cm")
    print(f"   Start: {env['start']}")
    print(f"   Goal: {env['goal']}")
    print(f"   Storage areas: {len(env['obstacles_rect'])}")
    
    # Hiển thị bản đồ
    print("\n🗺️  Displaying warehouse map...")
    plot_warehouse_map(env, show_grid=True)
    
    # Tham số
    params = dict(
        n_particles=40,
        max_iter=120,
        v_max=25.0,
        T0=2000.0,
        Tmin=0.1,
        alpha=0.96,
        max_iter_per_temp=40,
        T0_hybrid=200.0,
        alpha_hybrid=0.96
    )
    
    results = {}
    
    # SA
    print("\n" + "="*70)
    print("🔥 Running Simulated Annealing...")
    print("="*70)
    start_time = time.time()
    sa_path, sa_cost = sa_pathplanning(env, params)
    sa_time = time.time() - start_time
    results['SA'] = {'cost': sa_cost, 'time': sa_time, 'path': sa_path}
    print(f"\n✅ SA finished in {sa_time:.2f}s with cost = {sa_cost:.2f}\n")
    plot_result(env, sa_path, "Simulated Annealing Path", sa_cost)
    
    # PSO
    print("\n" + "="*70)
    print("🐝 Running PSO...")
    print("="*70)
    start_time = time.time()
    pso_path, pso_cost = pso_pathplanning(env, params)
    pso_time = time.time() - start_time
    results['PSO'] = {'cost': pso_cost, 'time': pso_time, 'path': pso_path}
    print(f"\n✅ PSO finished in {pso_time:.2f}s with cost = {pso_cost:.2f}\n")
    plot_result(env, pso_path, "PSO Path", pso_cost)
    
    # Hybrid
    print("\n" + "="*70)
    print("⚡ Running Hybrid PSO-SA...")
    print("="*70)
    start_time = time.time()
    hybrid_path, hybrid_cost = hybrid_pso_sa(env, params)
    hybrid_time = time.time() - start_time
    results['Hybrid'] = {'cost': hybrid_cost, 'time': hybrid_time, 'path': hybrid_path}
    print(f"\n✅ Hybrid finished in {hybrid_time:.2f}s with cost = {hybrid_cost:.2f}\n")
    plot_result(env, hybrid_path, "Hybrid PSO-SA Path", hybrid_cost)
    
    # So sánh
    print("\n" + "="*70)
    print("📊 COMPARISON RESULTS")
    print("="*70)
    print(f"{'Algorithm':<20} {'Cost':>12} {'Time (s)':>12} {'Quality':>12}")
    print("-"*70)
    
    best_cost = min(r['cost'] for r in results.values())
    for name, data in results.items():
        quality = "★★★★★" if data['cost'] == best_cost else "★★★★☆" if data['cost'] < best_cost * 1.1 else "★★★☆☆"
        print(f"{name:<20} {data['cost']:>12.2f} {data['time']:>12.2f} {quality:>12}")
    
    print("="*70)
    print(f"🏆 Best algorithm: {min(results, key=lambda k: results[k]['cost'])}")
    print("="*70)