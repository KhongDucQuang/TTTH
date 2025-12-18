import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ===========================================================
# Sinh môi trường warehouse (CẬP NHẬT: THÊM PHÒNG NHỎ G & H)
# ===========================================================
def generate_warehouse_scenario():
    env = {}
    env["width"], env["height"] = 900, 500
    WALL_THICKNESS = 5

    env["walls_rect"] = [
        # --- TƯỜNG BAO ---
        [0, 0, WALL_THICKNESS, 500],
        [900 - WALL_THICKNESS, 0, WALL_THICKNESS, 500],
        [0, 0, 900, WALL_THICKNESS],
        [0, 500 - WALL_THICKNESS, 900, WALL_THICKNESS],

        # --- KHU VỰC CŨ (A, B, C, D) ---
        # Area A (Cửa dưới)
        [450, 325, WALL_THICKNESS, 175], 
        [650, 325, WALL_THICKNESS, 175], 
        [450, 325, 600 - 450, WALL_THICKNESS], 

        # Area B (Cửa phải)
        [450, 125, WALL_THICKNESS, 125], 
        [680, 125, WALL_THICKNESS, 200 - 125], 
        [450, 125, 230, WALL_THICKNESS], 
        [450, 250, 230, WALL_THICKNESS], 

        # Area C (Cửa phải)
        [450, 0, WALL_THICKNESS, 125], 
        [680, 0, WALL_THICKNESS, 60], 

        # Area D (Cửa trái)
        [750, 325, WALL_THICKNESS, 380 - 325],
        [750, 325, 150, WALL_THICKNESS], 
        [750, 450, 150, WALL_THICKNESS], 

        # --- KHU VỰC E (Cửa trái) ---
        [750, 200, WALL_THICKNESS, 260 - 200],
        [750, 200, 150, WALL_THICKNESS], # Dưới của E

        # --- KHU VỰC F (Cửa trái) ---
        [750, 0, WALL_THICKNESS, 140],

        # =================================================
        # THÊM PHÒNG NHỎ (ĐÃ MỞ RỘNG CỬA)
        # =================================================
        
        # === Phòng H nhỏ (Nằm trong E) ===
        # Tọa độ: x[820-880], y[220-300] -> Rộng 60
        [820, 220, WALL_THICKNESS, 80],  # Tường trái
        [880, 220, WALL_THICKNESS, 80],  # Tường phải
        [820, 220, 60, WALL_THICKNESS],  # Tường dưới
        
        # Tường trên của H (Cửa rộng 50: Hở từ 825 -> 875)
        [820, 300, 5, WALL_THICKNESS],   # Mép tường trái (chỉ còn 5 đơn vị)
        [875, 300, 5 + WALL_THICKNESS, WALL_THICKNESS], # Mép tường phải (chỉ còn 5 đơn vị)

        # === Phòng G nhỏ (Nằm trong F) ===
        # Tọa độ: x[820-880], y[20-100] -> Rộng 60
        [820, 20, WALL_THICKNESS, 80],   # Tường trái
        [880, 20, WALL_THICKNESS, 80],   # Tường phải
        [820, 20, 60, WALL_THICKNESS],   # Tường dưới
        
        # Tường trên của G (Cửa rộng 50: Hở từ 825 -> 875)
        [820, 100, 5, WALL_THICKNESS],   # Mép tường trái (chỉ còn 5 đơn vị)
        [875, 100, 5 + WALL_THICKNESS, WALL_THICKNESS], # Mép tường phải (chỉ còn 5 đơn vị)
    ]

    env["area_labels"] = [
        {'name': 'A', 'pos': (550, 415)},
        {'name': 'B', 'pos': (565, 188)},
        {'name': 'C', 'pos': (565, 63)},
        {'name': 'D', 'pos': (825, 388)},
        
        # Dời nhãn E và F ra ngoài một chút
        {'name': 'E', 'pos': (785, 260)}, 
        {'name': 'F', 'pos': (785, 110)},
        
        # Nhãn cho phòng mới
        {'name': 'H', 'pos': (850, 260)}, # Trong E
        {'name': 'G', 'pos': (850, 60)},  # Trong F
        
        {'name': 'Hall', 'pos': (225, 250)}, 
    ]

    # Start ở trong phòng G nhỏ
    env["start"] = np.array([850, 60]) 
    env["goal"] = np.array([100, 100]) 

    env["n_waypoints"] = 25 
    env["safety_margin_soft"] = 30.0
    env["safety_margin_hard"] = 15.0 

    env["weights"] = np.array([2.0, 20.0, 200000.0, 0.0]) 

    return env

def plot_warehouse_map(env, show_grid=False):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, env["width"])
    ax.set_ylim(0, env["height"])
    
    # Vẽ tường
    for wall_data in env["walls_rect"]:
        x, y, w, h = wall_data
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='darkslategrey')
        ax.add_patch(rect)

    # Vẽ tên khu vực
    for label_info in env["area_labels"]:
        color = 'red' if label_info['name'] in ['G', 'H'] else 'blue' # Nổi bật phòng mới
        ax.text(label_info['pos'][0], label_info['pos'][1], label_info['name'], 
                ha='center', va='center', color=color, fontweight='bold', fontsize=12)

    if show_grid: ax.grid(True, alpha=0.3)
    if "start" in env: ax.plot(*env["start"], 'go', ms=10, label='Start')
    if "goal" in env: ax.plot(*env["goal"], 'ro', ms=10, label='Goal')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Test nhanh nếu chạy trực tiếp file này
if __name__ == "__main__":
    env = generate_warehouse_scenario()
    plot_warehouse_map(env)