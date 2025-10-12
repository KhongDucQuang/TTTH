import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
# ===========================================================
# Sinh môi trường warehouse 
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