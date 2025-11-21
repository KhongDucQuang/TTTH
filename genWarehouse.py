import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ===========================================================
# Sinh môi trường warehouse (CỬA ĐÃ ĐƯỢC MỞ RỘNG)
# ===========================================================
def generate_warehouse_scenario():
    env = {}
    env["width"], env["height"] = 900, 500
    WALL_THICKNESS = 5

    # === CẤU HÌNH CỬA RỘNG HƠN (GAP ~ 60-80 đơn vị) ===
    # Để robot có safety_margin=20 (đường kính 40) đi lọt dễ dàng
    
    env["walls_rect"] = [
        # Tường bao
        [0, 0, WALL_THICKNESS, 500],
        [900 - WALL_THICKNESS, 0, WALL_THICKNESS, 500],
        [0, 0, 900, WALL_THICKNESS],
        [0, 500 - WALL_THICKNESS, 900, WALL_THICKNESS],

        # === Area A (Cửa dưới) ===
        [450, 325, WALL_THICKNESS, 175], # Trái
        [650, 325, WALL_THICKNESS, 175], # Phải
        [450, 325, 600 - 450, WALL_THICKNESS], # Dưới (Cửa rộng 50: 600->650)

        # === Area B (Cửa phải) ===
        [450, 125, WALL_THICKNESS, 125], # Trái
        # Cửa phải mở rộng: Tường từ 125->200 (Hở 200->250)
        [680, 125, WALL_THICKNESS, 200 - 125], 
        [450, 125, 230, WALL_THICKNESS], # Dưới
        [450, 250, 230, WALL_THICKNESS], # Trên

        # === Area C (Cửa phải) ===
        [450, 0, WALL_THICKNESS, 125], # Trái
        # Cửa phải mở rộng: Tường từ 0->60 (Hở 60->125)
        [680, 0, WALL_THICKNESS, 60], 

        # === Area D (Cửa trái) ===
        # Cửa trái mở rộng: Tường từ 325->380 (Hở 380->450)
        [750, 325, WALL_THICKNESS, 380 - 325],
        [750, 325, 150, WALL_THICKNESS], # Dưới
        [750, 450, 150, WALL_THICKNESS], # Trên

        # === Area E (Cửa trái) ===
        # Cửa trái mở rộng: Tường từ 200->260 (Hở 260->325)
        [750, 200, WALL_THICKNESS, 260 - 200],
        [750, 200, 150, WALL_THICKNESS], # Dưới

        # === Area F (Cửa trái - ĐIỂM ĐẾN) ===
        # Cửa trái mở rộng: Tường từ 0->140 (Hở 140->200) - Rộng 60 đơn vị
        [750, 0, WALL_THICKNESS, 140],
    ]

    env["area_labels"] = [
        {'name': 'A', 'pos': (550, 415)},
        {'name': 'B', 'pos': (565, 188)},
        {'name': 'C', 'pos': (565, 63)},
        {'name': 'D', 'pos': (825, 388)},
        {'name': 'E', 'pos': (825, 263)},
        {'name': 'F', 'pos': (825, 100)},
        {'name': 'G', 'pos': (225, 250)},
    ]

    env["start"] = np.array([50, 250])
    env["goal"] = np.array([825, 100]) 

    env["n_waypoints"] = 300 # Tăng số điểm để đường đi mềm dẻo hơn
    env["safety_margin_soft"] = 30.0
    env["safety_margin_hard"] = 15.0 # Giữ margin 15, cửa rộng 60 là đi lọt

    # Trọng số: Ưu tiên độ dài và an toàn
    env["weights"] = np.array([2.0, 20.0, 200000.0, 0.0]) 

    return env

def plot_warehouse_map(env, show_grid=False):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, env["width"])
    ax.set_ylim(0, env["height"])
    
    for wall_data in env["walls_rect"]:
        x, y, w, h = wall_data
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='darkslategrey')
        ax.add_patch(rect)

    for label_info in env["area_labels"]:
        ax.text(label_info['pos'][0], label_info['pos'][1], f"Area {label_info['name']}", 
                ha='center', color='blue', fontweight='bold')

    if show_grid: ax.grid(True, alpha=0.3)
    if "start" in env: ax.plot(*env["start"], 'go', ms=10, label='Start')
    if "goal" in env: ax.plot(*env["goal"], 'ro', ms=10, label='Goal')
    
    plt.tight_layout()
    plt.show()