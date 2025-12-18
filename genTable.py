import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# ==============================================================================
# 1. BỘ HÀM BENCHMARK (f1 - f7)
# ==============================================================================
class Benchmarks:
    def f1_rosenbrock(x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
    
    def f2_ackley(x):
        d = len(x)
        return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(2*np.pi*x))/d) + 20 + np.e
    
    def f3_levy(x):
        w = 1 + (x - 1) / 4
        d = len(x)
        term1 = (np.sin(np.pi*w[0]))**2
        sum_mid = np.sum((w[:-1]-1)**2 * (1+10*np.sin(np.pi*w[:-1]+1)**2))
        term3 = (w[-1]-1)**2 * (1+np.sin(2*np.pi*w[-1])**2)
        return term1 + sum_mid + term3
    
    def f4_sphere(x):
        return np.sum(x**2)
    
    def f5_sum_squares(x):
        return np.sum([(i+1)*x[i]**2 for i in range(len(x))])
    
    def f6_zakharov(x):
        s2 = np.sum([0.5*(i+1)*x[i] for i in range(len(x))])
        return np.sum(x**2) + s2**2 + s2**4
    
    def f7_michalewicz(x):
        m = 10
        return -np.sum(np.sin(x) * (np.sin(((np.arange(len(x))+1) * x**2)/np.pi))**(2*m))

# ==============================================================================
# 2. CÁC THUẬT TOÁN (ĐÃ CẬP NHẬT TRẢ VỀ SỐ VÒNG LẶP HỘI TỤ)
# ==============================================================================

def run_sa_yours(func, dim, bounds, max_iter):
    start_time = time.time()
    x = np.random.uniform(bounds[0], bounds[1], dim)
    cost = func(x)
    best_cost = cost
    last_iter = 0
    
    T = 100.0; alpha = 0.95
    
    for t in range(max_iter):
        scale = (bounds[1]-bounds[0]) * 0.1 * (T/100.0)
        new_x = x + np.random.normal(0, scale, dim)
        new_x = np.clip(new_x, bounds[0], bounds[1])
        new_cost = func(new_x)
        
        delta = new_cost - cost
        if delta < 0 or np.random.rand() < np.exp(-delta/(T+1e-10)):
            x, cost = new_x, new_cost
            if cost < best_cost: 
                best_cost = cost
                last_iter = t # Ghi nhận vòng lặp cải thiện
        T *= alpha
        
    return best_cost, last_iter, time.time() - start_time

def run_pso_yours(func, dim, bounds, max_iter):
    start_time = time.time()
    n_part = 30
    X = np.random.uniform(bounds[0], bounds[1], (n_part, dim))
    V = np.zeros_like(X)
    P = X.copy()
    P_val = np.array([func(i) for i in X])
    
    g_idx = np.argmin(P_val)
    g_pos = P[g_idx].copy()
    g_val = P_val[g_idx]
    last_iter = 0
    
    w_max, w_min = 0.9, 0.4; c1, c2 = 2.0, 2.0
    
    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter
        r1, r2 = np.random.rand(n_part, dim), np.random.rand(n_part, dim)
        V = w*V + c1*r1*(P-X) + c2*r2*(g_pos-X)
        X = np.clip(X + V, bounds[0], bounds[1])
        
        vals = np.array([func(i) for i in X])
        mask = vals < P_val
        P[mask] = X[mask]; P_val[mask] = vals[mask]
        
        curr_min = np.min(vals)
        if curr_min < g_val:
            g_val = curr_min
            g_pos = X[np.argmin(vals)].copy()
            last_iter = t # Ghi nhận vòng lặp cải thiện
            
    return g_val, last_iter, time.time() - start_time

def run_hybrid_yours(func, dim, bounds, max_iter):
    start_time = time.time()
    n_part = 30
    X = np.random.uniform(bounds[0], bounds[1], (n_part, dim))
    V = np.zeros_like(X)
    P = X.copy()
    P_val = np.array([func(i) for i in X])
    
    g_idx = np.argmin(P_val)
    g_pos = P[g_idx].copy()
    g_val = P_val[g_idx]
    last_iter = 0
    
    w_max, w_min = 0.9, 0.4; c1, c2 = 2.0, 2.0
    T = 100.0; alpha = 0.95; L_sa = 5
    
    for t in range(max_iter):
        # 1. PSO Phase
        w = w_max - (w_max - w_min) * t / max_iter
        r1, r2 = np.random.rand(n_part, dim), np.random.rand(n_part, dim)
        V = w*V + c1*r1*(P-X) + c2*r2*(g_pos-X)
        X = np.clip(X + V, bounds[0], bounds[1])
        
        vals = np.array([func(i) for i in X])
        mask = vals < P_val
        P[mask] = X[mask]; P_val[mask] = vals[mask]
        
        if np.min(P_val) < g_val:
            g_val = np.min(P_val)
            g_pos = P[np.argmin(P_val)].copy()
            last_iter = t
            
        # 2. SA Phase on gBest
        scale = (bounds[1]-bounds[0]) * 0.05 * (T/100.0)
        for _ in range(L_sa):
            candidate = g_pos + np.random.normal(0, scale, dim)
            candidate = np.clip(candidate, bounds[0], bounds[1])
            c_cost = func(candidate)
            delta = c_cost - g_val
            
            if delta < 0:
                g_pos = candidate
                g_val = c_cost
                last_iter = t # Ghi nhận nếu SA tìm được điểm tốt hơn
            elif np.random.rand() < np.exp(-delta/(T+1e-10)):
                g_pos = candidate
                g_val = c_cost
        
        T *= alpha
    return g_val, last_iter, time.time() - start_time

# --- CÁC THUẬT TOÁN ĐỐI THỦ (CẬP NHẬT) ---

def run_hs(func, dim, bounds, max_iter):
    start_time = time.time()
    HMS = 30; HMCR = 0.9; PAR = 0.3; bw = 0.01 * (bounds[1] - bounds[0])
    HM = np.random.uniform(bounds[0], bounds[1], (HMS, dim))
    fitness = np.array([func(x) for x in HM])
    best_val = np.min(fitness)
    last_iter = 0
    
    for t in range(max_iter):
        new_h = np.zeros(dim)
        for i in range(dim):
            if np.random.rand() < HMCR:
                idx = np.random.randint(0, HMS)
                new_h[i] = HM[idx, i]
                if np.random.rand() < PAR:
                    new_h[i] += np.random.uniform(-bw, bw)
            else:
                new_h[i] = np.random.uniform(bounds[0], bounds[1])
        new_h = np.clip(new_h, bounds[0], bounds[1])
        new_fit = func(new_h)
        
        worst_idx = np.argmax(fitness)
        if new_fit < fitness[worst_idx]:
            HM[worst_idx] = new_h
            fitness[worst_idx] = new_fit
            
        curr_min = np.min(fitness)
        if curr_min < best_val:
            best_val = curr_min
            last_iter = t
            
    return best_val, last_iter, time.time() - start_time

def run_fa(func, dim, bounds, max_iter):
    start_time = time.time()
    n_fireflies = 30; alpha = 0.5; beta0 = 1.0; gamma = 1.0
    X = np.random.uniform(bounds[0], bounds[1], (n_fireflies, dim))
    I = np.array([func(x) for x in X])
    best_val = np.min(I)
    last_iter = 0
    
    for t in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if I[j] < I[i]:
                    r = np.linalg.norm(X[i] - X[j])
                    beta = beta0 * np.exp(-gamma * r**2)
                    X[i] += beta * (X[j] - X[i]) + alpha * (np.random.rand(dim) - 0.5)
                    X[i] = np.clip(X[i], bounds[0], bounds[1])
                    I[i] = func(X[i])
                    if I[i] < best_val: 
                        best_val = I[i]
                        last_iter = t
        alpha *= 0.97
    return best_val, last_iter, time.time() - start_time

def run_abc(func, dim, bounds, max_iter):
    start_time = time.time()
    n_pop = 30; limit = 100
    X = np.random.uniform(bounds[0], bounds[1], (n_pop, dim))
    fit = np.array([func(x) for x in X])
    trial = np.zeros(n_pop)
    best_val = np.min(fit)
    last_iter = 0
    
    for t in range(max_iter):
        for i in range(n_pop): # Employed
            k = np.random.randint(0, n_pop)
            while k == i: k = np.random.randint(0, n_pop)
            phi = np.random.uniform(-1, 1, dim)
            v = X[i] + phi * (X[i] - X[k])
            v = np.clip(v, bounds[0], bounds[1])
            f_v = func(v)
            if f_v < fit[i]:
                X[i], fit[i], trial[i] = v, f_v, 0
            else:
                trial[i] += 1
        
        # Scout
        idx_max = np.argmax(trial)
        if trial[idx_max] > limit:
            X[idx_max] = np.random.uniform(bounds[0], bounds[1], dim)
            fit[idx_max] = func(X[idx_max])
            trial[idx_max] = 0
            
        curr_min = np.min(fit)
        if curr_min < best_val: 
            best_val = curr_min
            last_iter = t
    return best_val, last_iter, time.time() - start_time

def run_ga(func, dim, bounds, max_iter):
    start_time = time.time()
    n_pop = 30; pc = 0.8; pm = 0.1
    pop = np.random.uniform(bounds[0], bounds[1], (n_pop, dim))
    fit = np.array([func(x) for x in pop])
    best_val = np.min(fit)
    last_iter = 0
    
    for t in range(max_iter):
        new_pop = []
        for _ in range(n_pop): # Selection
            i1, i2 = np.random.randint(0, n_pop, 2)
            p1 = pop[i1] if fit[i1] < fit[i2] else pop[i2]
            new_pop.append(p1)
        new_pop = np.array(new_pop)
        
        for i in range(0, n_pop, 2): # Crossover & Mutation
            if np.random.rand() < pc:
                c = np.random.rand()
                new_pop[i] = c*new_pop[i] + (1-c)*new_pop[i+1]
            if np.random.rand() < pm:
                new_pop[i] += np.random.normal(0, 1.0, dim)
        
        pop = np.clip(new_pop, bounds[0], bounds[1])
        fit = np.array([func(x) for x in pop])
        
        curr_min = np.min(fit)
        if curr_min < best_val:
            best_val = curr_min
            last_iter = t
            
    return best_val, last_iter, time.time() - start_time

# ==============================================================================
# 3. CHẠY VÀ XUẤT BÁO CÁO (MAIN)
# ==============================================================================
if __name__ == "__main__":
    # Settings
    N_RUNS = 20
    DIM = 30
    ITER = 500
    
    funcs = [
        ("f1 (Rosenbrock)", Benchmarks.f1_rosenbrock, [-30, 30]),
        ("f2 (Ackley)",     Benchmarks.f2_ackley,     [-32, 32]),
        ("f3 (Levy)",       Benchmarks.f3_levy,       [-10, 10]),
        ("f4 (Sphere)",     Benchmarks.f4_sphere,     [-100, 100]),
        ("f5 (SumSquare)",  Benchmarks.f5_sum_squares,[-10, 10]),
        ("f6 (Zakharov)",   Benchmarks.f6_zakharov,   [-5, 10]),
        ("f7 (Michalewicz)",Benchmarks.f7_michalewicz,[0, np.pi])
    ]
    
    algos = [
        ("HS", run_hs),
        ("FA", run_fa),
        ("ABC", run_abc),
        ("GA", run_ga),
        ("Your_SA", run_sa_yours),
        ("Your_PSO", run_pso_yours),
        ("Your_Hybrid", run_hybrid_yours)
    ]
    
    print(f"Running Benchmark ({N_RUNS} runs per algo)...")
    results = []
    
    for f_name, f_func, f_bounds in funcs:
        print(f"Processing {f_name}...")
        row = {"Function": f_name}
        for a_name, a_func in algos:
            fit_vals = []
            iter_vals = []
            times = []
            
            for _ in tqdm(range(N_RUNS), desc=f"  {a_name}", leave=False):
                v, it, tm = a_func(f_func, DIM, f_bounds, ITER)
                fit_vals.append(v)
                iter_vals.append(it)
                times.append(tm)
            
            # Ghi nhận số liệu: Best Fitness, Mean Fitness, Mean Iteration
            row[f"{a_name}_BestFit"] = np.min(fit_vals)
            row[f"{a_name}_AvgFit"] = np.mean(fit_vals)
            row[f"{a_name}_MeanIter"] = int(np.mean(iter_vals))
            # row[f"{a_name}_Time"] = np.mean(times) # Bỏ comment nếu muốn xem thời gian
            
        results.append(row)
        
    df = pd.DataFrame(results)
    pd.options.display.float_format = '{:,.2e}'.format
    
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE (Fitness & Iterations)")
    print("="*80)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df)
    
    df.to_csv("full_comparison_final.csv")
    print("\nSaved detailed results to 'full_comparison_final.csv'")