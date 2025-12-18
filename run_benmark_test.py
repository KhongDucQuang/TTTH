import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================================================================
# 1. ĐỊNH NGHĨA 7 HÀM TEST (BENCHMARK FUNCTIONS)
# ==============================================================================
class BenchmarkFunctions:
    def __init__(self):
        self.m_michalewicz = 10 

    def f1_rosenbrock(self, x):
        sum_val = 0
        dim = len(x)
        for i in range(dim - 1):
            sum_val += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        return sum_val

    def f2_ackley(self, x):
        dim = len(x)
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / dim)
        return term1 + term2 + 20 + np.e

    def f3_levy(self, x):
        dim = len(x)
        w = 1 + (x - 1) / 4
        term1 = (np.sin(np.pi * w[0]))**2
        term3 = (w[dim-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[dim-1]))**2)
        sum_mid = np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2))
        return term1 + sum_mid + term3

    def f4_sphere(self, x):
        return np.sum(x**2)

    def f5_sum_squares(self, x):
        sum_val = 0
        for i in range(len(x)):
            sum_val += (i + 1) * (x[i]**2)
        return sum_val

    def f6_zakharov(self, x):
        sum1 = np.sum(x**2)
        sum2 = np.sum([0.5 * (i + 1) * x[i] for i in range(len(x))])
        return sum1 + sum2**2 + sum2**4

    def f7_michalewicz(self, x):
        m = self.m_michalewicz
        sum_val = 0
        for i in range(len(x)):
            sum_val += np.sin(x[i]) * (np.sin(((i + 1) * x[i]**2) / np.pi))**(2 * m)
        return -sum_val

# ==============================================================================
# 2. CÁC THUẬT TOÁN (PHIÊN BẢN TOÁN HỌC)
# ==============================================================================

# --- SA (Simulated Annealing) ---
def run_sa(func, dim, bounds, max_iter):
    # Khởi tạo
    current_pos = np.random.uniform(bounds[0], bounds[1], dim)
    current_cost = func(current_pos)
    
    best_pos = current_pos.copy()
    best_cost = current_cost
    
    T = 100.0
    alpha = 0.95
    
    history = []
    
    for i in range(max_iter):
        # Rung lắc nhẹ (Scale giảm dần theo thời gian)
        scale = (bounds[1] - bounds[0]) * 0.1 * (T / 100.0)
        new_pos = current_pos + np.random.normal(0, scale, dim)
        new_pos = np.clip(new_pos, bounds[0], bounds[1])
        
        new_cost = func(new_pos)
        
        delta = new_cost - current_cost
        
        # Chấp nhận?
        if delta < 0 or np.random.rand() < np.exp(-delta / (T + 1e-10)):
            current_pos = new_pos
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos = new_pos
        
        # Giảm nhiệt
        T = T * alpha
        history.append(best_cost)
        
    return best_cost, history

# --- PSO (Particle Swarm Optimization) ---
def run_pso(func, dim, bounds, max_iter, n_particles=30):
    X = np.random.uniform(bounds[0], bounds[1], (n_particles, dim))
    V = np.zeros_like(X)
    
    P_best = X.copy()
    P_best_val = np.array([func(x) for x in X])
    
    g_best_idx = np.argmin(P_best_val)
    g_best = P_best[g_best_idx].copy()
    g_best_val = P_best_val[g_best_idx]
    
    w_max, w_min = 0.9, 0.4
    c1, c2 = 2.0, 2.0
    
    history = []
    
    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter
        
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        
        V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (g_best - X)
        X = X + V
        X = np.clip(X, bounds[0], bounds[1])
        
        # Đánh giá
        vals = np.array([func(x) for x in X])
        
        # Cập nhật pBest
        better_mask = vals < P_best_val
        P_best[better_mask] = X[better_mask]
        P_best_val[better_mask] = vals[better_mask]
        
        # Cập nhật gBest
        min_idx = np.argmin(P_best_val)
        if P_best_val[min_idx] < g_best_val:
            g_best_val = P_best_val[min_idx]
            g_best = P_best[min_idx].copy()
            
        history.append(g_best_val)
        
    return g_best_val, history

# --- Hybrid PSO-SA ---
def run_hybrid(func, dim, bounds, max_iter, n_particles=30):
    X = np.random.uniform(bounds[0], bounds[1], (n_particles, dim))
    V = np.zeros_like(X)
    
    P_best = X.copy()
    P_best_val = np.array([func(x) for x in X])
    
    g_best_idx = np.argmin(P_best_val)
    g_best = P_best[g_best_idx].copy()
    g_best_val = P_best_val[g_best_idx]
    
    w_max, w_min = 0.9, 0.4
    c1, c2 = 2.0, 2.0
    
    T = 100.0
    alpha = 0.95
    
    history = []
    
    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter
        
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        
        V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (g_best - X)
        X = X + V
        X = np.clip(X, bounds[0], bounds[1])
        
        vals = np.array([func(x) for x in X])
        
        # --- HYBRID LOGIC ---
        for i in range(n_particles):
            delta = vals[i] - P_best_val[i]
            # SA condition: Tốt hơn hoặc may mắn
            if delta < 0 or np.random.rand() < np.exp(-delta / (T + 1e-10)):
                P_best[i] = X[i].copy()
                P_best_val[i] = vals[i]
                
                if vals[i] < g_best_val:
                    g_best_val = vals[i]
                    g_best = X[i].copy()
                    
        T *= alpha
        history.append(g_best_val)
        
    return g_best_val, history

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH (MAIN RUNNER)
# ==============================================================================
if __name__ == "__main__":
    bench = BenchmarkFunctions()
    
    # Danh sách các hàm cần test
    functions = [
        ("F1: Rosenbrock", bench.f1_rosenbrock, [-30, 30]),
        ("F2: Ackley",     bench.f2_ackley,     [-32, 32]),
        ("F3: Levy",       bench.f3_levy,       [-10, 10]),
        ("F4: Sphere",     bench.f4_sphere,     [-100, 100]),
        ("F5: Sum Squares",bench.f5_sum_squares,[-10, 10]),
        ("F6: Zakharov",   bench.f6_zakharov,   [-5, 10]),
        ("F7: Michalewicz",bench.f7_michalewicz,[0, np.pi]) 
    ]
    
    dim = 30           # Số chiều (Dimension) - Bài báo thường dùng 30 hoặc 50
    max_iter = 500     # Số vòng lặp
    runs = 1           # Số lần chạy trung bình (Để test nhanh để 1, chuẩn thì để 30)

    print(f"{'Function':<20} | {'Algo':<10} | {'Best Cost':<15} | {'Time(s)':<10}")
    print("-" * 65)

    # Dictionary lưu lịch sử để vẽ đồ thị (Chỉ lấy F1 và F2 làm mẫu)
    plot_data = {} 

    for name, func, bounds in functions:
        # Chạy SA
        start = time.time()
        sa_val, sa_hist = run_sa(func, dim, bounds, max_iter)
        print(f"{name:<20} | SA         | {sa_val:.4e}      | {time.time()-start:.4f}")
        
        # Chạy PSO
        start = time.time()
        pso_val, pso_hist = run_pso(func, dim, bounds, max_iter)
        print(f"{name:<20} | PSO        | {pso_val:.4e}      | {time.time()-start:.4f}")
        
        # Chạy Hybrid
        start = time.time()
        hyb_val, hyb_hist = run_hybrid(func, dim, bounds, max_iter)
        print(f"{name:<20} | Hybrid     | {hyb_val:.4e}      | {time.time()-start:.4f}")
        print("-" * 65)

        # Lưu dữ liệu vẽ biểu đồ cho F1 và F2
        if "Rosenbrock" in name or "Ackley" in name:
            plot_data[name] = (sa_hist, pso_hist, hyb_hist)

    # --- VẼ BIỂU ĐỒ HỘI TỤ (CONVERGENCE PLOT) ---
    print("\nĐang vẽ biểu đồ so sánh hội tụ...")
    plt.figure(figsize=(14, 6))
    
    idx = 1
    for func_name, (h_sa, h_pso, h_hyb) in plot_data.items():
        plt.subplot(1, 2, idx)
        plt.plot(h_sa, label='SA', linestyle='--', color='green')
        plt.plot(h_pso, label='PSO', linestyle='-.', color='blue')
        plt.plot(h_hyb, label='Hybrid PSO-SA', linewidth=2, color='red')
        
        plt.title(f"Convergence: {func_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Cost Value (Log scale)")
        plt.yscale("log") # Dùng thang log để nhìn rõ sự khác biệt khi về 0
        plt.legend()
        plt.grid(True, alpha=0.3)
        idx += 1
        
    plt.tight_layout()
    plt.show()