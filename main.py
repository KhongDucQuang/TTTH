import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from genWarehouse import generate_warehouse_scenario, plot_warehouse_map
from costFuncAndUtilities import collision_cost, cost_function, smoothness_cost, path_length, perturb_path, random_path, interpolate_path
from hybrid_SA_PSO import hybrid_pso_sa
from pso import pso_pathplanning
from sa import sa_pathplanning
# ===========================================================
# Hiển thị kết quả
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
# Main test
# ===========================================================
if __name__ == "__main__":
    print("="*70)
    print("WAREHOUSE PATH PLANNING - PSO, SA, HYBRID COMPARISON")
    print("="*70)
    
    # Tạo môi trường
    env = generate_warehouse_scenario()
    
    print("\nEnvironment Info:")
    print(f"   Size: {env['width']} x {env['height']} cm")
    print(f"   Start: {env['start']}")
    print(f"   Goal: {env['goal']}")
    print(f"   Storage areas: {len(env['obstacles_rect'])}")
    
    # Hiển thị bản đồ
    print("\nDisplaying warehouse map...")
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
    print("Running Simulated Annealing...")
    print("="*70)
    start_time = time.time()
    sa_path, sa_cost = sa_pathplanning(env, params)
    sa_time = time.time() - start_time
    results['SA'] = {'cost': sa_cost, 'time': sa_time, 'path': sa_path}
    print(f"\nSA finished in {sa_time:.2f}s with cost = {sa_cost:.2f}\n")
    plot_result(env, sa_path, "Simulated Annealing Path", sa_cost)
    
    # PSO
    print("\n" + "="*70)
    print("Running PSO...")
    print("="*70)
    start_time = time.time()
    pso_path, pso_cost = pso_pathplanning(env, params)
    pso_time = time.time() - start_time
    results['PSO'] = {'cost': pso_cost, 'time': pso_time, 'path': pso_path}
    print(f"\nPSO finished in {pso_time:.2f}s with cost = {pso_cost:.2f}\n")
    plot_result(env, pso_path, "PSO Path", pso_cost)
    
    # Hybrid
    print("\n" + "="*70)
    print("Running Hybrid PSO-SA...")
    print("="*70)
    start_time = time.time()
    hybrid_path, hybrid_cost = hybrid_pso_sa(env, params)
    hybrid_time = time.time() - start_time
    results['Hybrid'] = {'cost': hybrid_cost, 'time': hybrid_time, 'path': hybrid_path}
    print(f"\nHybrid finished in {hybrid_time:.2f}s with cost = {hybrid_cost:.2f}\n")
    plot_result(env, hybrid_path, "Hybrid PSO-SA Path", hybrid_cost)
    
    # So sánh
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Algorithm':<20} {'Cost':>12} {'Time (s)':>12} {'Quality':>12}")
    print("-"*70)
    
    best_cost = min(r['cost'] for r in results.values())
    for name, data in results.items():
        quality = "★★★★★" if data['cost'] == best_cost else "★★★★☆" if data['cost'] < best_cost * 1.1 else "★★★☆☆"
        print(f"{name:<20} {data['cost']:>12.2f} {data['time']:>12.2f} {quality:>12}")
    
    print("="*70)
    print(f"Best algorithm: {min(results, key=lambda k: results[k]['cost'])}")
    print("="*70)