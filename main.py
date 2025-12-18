import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Import các file của bạn
from genWarehouse import generate_warehouse_scenario, plot_warehouse_map
# Đảm bảo import hàm generate_bspline_path mới
from costFuncAndUtilities import cost_function, random_path, perturb_path, generate_bspline_path
from sa import sa_pathplanning
from pso import pso_pathplanning
from hybrid_SA_PSO import hybrid_pso_sa

# ===========================================================
# Hiển thị kết quả (Cập nhật để vẽ B-Spline)
# ===========================================================
def plot_result(env, full_path, title, cost):
    """
    Vẽ kết quả đường đi trên bản đồ, bao gồm cả đường cong B-spline mượt.
    'full_path' là đường đi hoàn chỉnh [start, wp1, ..., goal].
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, env["width"])
    ax.set_ylim(0, env["height"])
    ax.set_xlabel('X (cm)', fontsize=11)
    ax.set_ylabel('Y (cm)', fontsize=11)

    # Vẽ các bức tường
    for wall_data in env["walls_rect"]:
        x, y, w, h = wall_data
        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                 edgecolor='black', facecolor='darkslategrey')
        ax.add_patch(rect)
        
    # Thêm tên khu vực
    for label_info in env["area_labels"]:
        name, pos = label_info['name'], label_info['pos']
        ax.text(pos[0], pos[1], f'Area {name}', ha='center', va='center', 
                fontsize=13, color='blue', fontweight='bold')

    # 1. TẠO VÀ VẼ ĐƯỜNG CONG B-SPLINE MƯỢT
    smooth_path = generate_bspline_path(full_path, n_points=200, degree=3)
    if smooth_path.size > 0:
        ax.plot(smooth_path[:, 0], smooth_path[:, 1], 'b-', linewidth=3, 
                label='Smoothed Path (B-spline)', zorder=5, alpha=0.9)

    # 2. (Tùy chọn) Vẽ các waypoints và đường nối thẳng để tham khảo
    ax.plot(full_path[:, 0], full_path[:, 1], '--o', color='cyan', markersize=8, 
            linewidth=1.5, zorder=6, label='Waypoints')

    # Vẽ điểm Start và Goal
    ax.plot(*env["start"], 'go', markersize=14, label='Start', zorder=10, mec='black')
    ax.plot(*env["goal"], 'ro', markersize=14, label='Goal', zorder=10, mec='black')
    
    full_title = f"{title}\nFinal Cost: {cost:.2f}"
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# ===========================================================
# Main
# ===========================================================
if __name__ == "__main__":
    print("="*70)
    print("WAREHOUSE PATH PLANNING - PSO, SA, HYBRID COMPARISON")
    print("="*70)
    
    env = generate_warehouse_scenario()
    
    print("\nEnvironment Info:")
    print(f"   Size: {env['width']} x {env['height']} cm")
    print(f"   Start: {env['start']}")
    print(f"   Goal: {env['goal']}")
    print(f"   Waypoints to optimize: {env['n_waypoints']}")
    
    plot_warehouse_map(env, show_grid=True)
    
    # Tinh chỉnh tham số để có kết quả tốt hơn
    params = dict(
        n_particles=100,
        max_iter=200,
        v_max=1.0,
        T0=200.0,
        Tmin=0.01,
        alpha=0.97,
        max_iter_per_temp=50,
        T0_hybrid=100.0,
        alpha_hybrid=0.97
    )
    
    results = {}
    
    # Chạy SA
    print("\n" + "="*70 + "\nRunning Simulated Annealing...\n" + "="*70)
    start_time = time.time()
    sa_path, sa_cost = sa_pathplanning(env, params)
    sa_time = time.time() - start_time
    results['SA'] = {'cost': sa_cost, 'time': sa_time, 'path': sa_path}
    print(f"\nSA finished in {sa_time:.2f}s with cost = {sa_cost:.2f}\n")
    plot_result(env, sa_path, "Simulated Annealing Path", sa_cost)
    
    # Chạy PSO
    print("\n" + "="*70 + "\nRunning PSO...\n" + "="*70)
    start_time = time.time()
    pso_path, pso_cost = pso_pathplanning(env, params)
    pso_time = time.time() - start_time
    results['PSO'] = {'cost': pso_cost, 'time': pso_time, 'path': pso_path}
    print(f"\nPSO finished in {pso_time:.2f}s with cost = {pso_cost:.2f}\n")
    plot_result(env, pso_path, "PSO Path", pso_cost)
    
    # Chạy Hybrid
    print("\n" + "="*70 + "\nRunning Hybrid PSO-SA...\n" + "="*70)
    start_time = time.time()
    hybrid_path, hybrid_cost = hybrid_pso_sa(env, params)
    hybrid_time = time.time() - start_time
    results['Hybrid'] = {'cost': hybrid_cost, 'time': hybrid_time, 'path': hybrid_path}
    print(f"\nHybrid finished in {hybrid_time:.2f}s with cost = {hybrid_cost:.2f}\n")
    plot_result(env, hybrid_path, "Hybrid PSO-SA Path", hybrid_cost)
    
    # In bảng so sánh kết quả
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Algorithm':<20} {'Final Cost':>15} {'Time (s)':>15}")
    print("-"*70)
    
    # Sắp xếp kết quả theo chi phí từ thấp đến cao
    sorted_results = sorted(results.items(), key=lambda item: item[1]['cost'])
    
    for name, data in sorted_results:
        print(f"{name:<20} {data['cost']:>15.2f} {data['time']:>15.2f}")
    
    print("="*70)
    if sorted_results:
        best_algo = sorted_results[0][0]
        print(f"Best algorithm based on cost: {best_algo}")
    print("="*70)