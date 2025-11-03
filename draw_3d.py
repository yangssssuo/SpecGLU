import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import io
from PIL import Image

# 读取数据 - 重新分配坐标轴
data = pd.read_csv('data/merged_file_rtp.csv')
x = data['esp2']  # ESP方差
y = data['homo_lumo_gap'] * 27.2114  # y保持不变
z = data['delta_est']  # 原来的x改为z
smiles = data['smiles']

# 设置科研论文风格
plt.style.use('default')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 22,
    'axes.titlesize': 20,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 1,
    'figure.titlesize': 20,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

from scipy.spatial import ConvexHull

def find_corner_points(x_data, y_data, smiles_data):
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    
    # 创建点的集合
    points = np.column_stack((x_array, y_array))
    
    # 计算凸包
    hull = ConvexHull(points)
    hull_indices = hull.vertices
    hull_points = points[hull_indices]
    
    # 找到数据的边界
    x_min, x_max = np.min(x_array), np.max(x_array)
    y_min, y_max = np.min(y_array), np.max(y_array)
    
    # 初始化四个角落的点
    extremes = {}
    min_distances = {
        'left_bottom': float('inf'),
        'left_top': float('inf'),
        'right_bottom': float('inf'),
        'right_top': float('inf')
    }
    
    # 定义四个角落的理想位置
    corners = {
        'left_bottom': (x_min, y_min),
        'left_top': (x_min, y_max),
        'right_bottom': (x_max, y_min),
        'right_top': (x_max, y_max)
    }
    
    # 在凸包顶点中寻找最接近四个角落的点
    for hull_idx in hull_indices:
        point = points[hull_idx]
        
        for corner_name, corner_pos in corners.items():
            # 计算到理想角落的距离（归一化）
            dx = (point[0] - corner_pos[0]) / (x_max - x_min)
            dy = (point[1] - corner_pos[1]) / (y_max - y_min)
            distance = np.sqrt(dx**2 + dy**2)
            
            # 如果这个点更接近这个角落，就更新
            if distance < min_distances[corner_name]:
                min_distances[corner_name] = distance
                extremes[corner_name] = hull_idx
    
    # 确保四个角落都找到了点，如果没有找到，使用备选方案
    for corner_name in corners.keys():
        if corner_name not in extremes:
            # 备选方案：在所有点中找到最接近的
            corner_pos = corners[corner_name]
            distances = np.sqrt(((x_array - corner_pos[0]) / (x_max - x_min))**2 + 
                              ((y_array - corner_pos[1]) / (y_max - y_min))**2)
            extremes[corner_name] = np.argmin(distances)
    
    return extremes


def find_global_extremes(x_data, y_data, z_data, smiles_data):
    """找到6个全局极值点"""
    extremes = {}
    
    # 找到每个维度的最小值和最大值
    extremes['x_min'] = np.argmin(x_data)  # ESP方差最小
    extremes['x_max'] = np.argmax(x_data)  # ESP方差最大
    extremes['y_min'] = np.argmin(y_data)  # HOMO-LUMO Gap最小
    extremes['y_max'] = np.argmax(y_data)  # HOMO-LUMO Gap最大
    extremes['z_min'] = np.argmin(z_data)  # Delta Est最小
    extremes['z_max'] = np.argmax(z_data)  # Delta Est最大
    
    return extremes

def find_farthest_points_3d(x_data, y_data, z_data, smiles_data, n_points=4):
    """找到距离3D中心最远的几个点"""
    # 计算数据中心
    x_center = np.mean(x_data)
    y_center = np.mean(y_data)
    z_center = np.mean(z_data)
    
    # 计算每个点到中心的距离
    distances = np.sqrt((x_data - x_center)**2 + 
                       (y_data - y_center)**2 + 
                       (z_data - z_center)**2)
    
    # 找到距离最远的n个点
    farthest_indices = np.argsort(distances)[-n_points:]
    
    return farthest_indices

def draw_molecule(smiles_str, size=(150, 150)):
    """绘制分子结构"""
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is not None:
            img = Draw.MolToImage(mol, size=size, kekulize=True)
            return img
    except:
        pass
    return None

def get_safe_offset(ax, x_pos, y_pos, offset_base=80, used_positions=None):
    if used_positions is None:
        used_positions = []
        
    # 候选偏移方向（8个方向）
    candidate_offsets = [
        (0, offset_base), (0, -offset_base),
        (offset_base, 0), (-offset_base, 0),
        (offset_base, offset_base), (-offset_base, -offset_base),
        (-offset_base, offset_base), (offset_base, -offset_base),
    ]

    # 逐步扩大搜索半径
    for scale in [1, 1.5, 2,2.5]:
        for dx, dy in candidate_offsets:
            offset = (dx * scale, dy * scale)
            offset_pos = (x_pos + offset[0], y_pos + offset[1])
            # 与已用位置比较距离
            is_clear = True
            for used in used_positions:
                dist = np.sqrt((offset_pos[0] - used[0])**2 + (offset_pos[1] - used[1])**2)
                if dist < offset_base * 1.2:
                    is_clear = False
                    break
            if is_clear:
                used_positions.append(offset_pos)
                return offset
    # 若无可用位置，默认强制使用最后一个扩大倍率
    fallback = (candidate_offsets[0][0] * 3, candidate_offsets[0][1] * 3)
    used_positions.append((x_pos + fallback[0], y_pos + fallback[1]))
    return fallback

def add_molecule_to_plot(ax, x_pos, y_pos, smiles_str, used_positions=None):
    """在图上添加分子结构，更靠近标记点，避免重叠"""
    mol_img = draw_molecule(smiles_str)
    mol_img.save('figs/mols/{}.png'.format(smiles_str+'_'+str(x_pos)+'_'+str(y_pos)))  # 保存分子图像
    if mol_img is not None:
        # 转换为matplotlib可用的格式
        # buf = io.BytesIO()
        # mol_img.save(buf, format='PNG')
        # buf.seek(0)
        # img = Image.open(buf)
        
        # # 获取安全的偏移位置
        # offset = get_safe_offset(ax, x_pos, y_pos, used_positions=used_positions)
        
        # # 添加到图上，使用白色背景覆盖
        # imagebox = OffsetImage(img, zoom=0.6)
        # ab = AnnotationBbox(imagebox, (x_pos, y_pos), 
        #                    xybox=offset, boxcoords="offset points",
        #                    frameon=True, pad=0.1,
        #                    bboxprops=dict(boxstyle="round,pad=0.1", 
        #                                 facecolor='white', 
        #                                 edgecolor='gray',
        #                                 alpha=0.95,
        #                                 linewidth=1))
        # ax.add_artist(ab)
        return True
    else:
        # 如果无法绘制分子，添加文本标签作为备选
        offset = get_safe_offset(ax, x_pos, y_pos, 40, used_positions)
        ax.annotate(f'Mol: {smiles_str[:10]}...', 
                   xy=(x_pos, y_pos), 
                   xytext=offset, textcoords='offset points',
                   fontsize=8, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                           edgecolor='black', alpha=0.8))
        return False


# 找到6个全局极值点
global_extremes = find_global_extremes(x, y, z, smiles)

# 创建单独的3D图
fig = plt.figure(figsize=(12, 8), dpi=600)
ax = fig.add_subplot(111, projection='3d')

# 使用参考图片的配色方案，现在颜色映射到z轴(delta_est)
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.7, 
                    edgecolors='none')

# 设置标签 - 更新坐标轴标签，增加z轴标签的显示
ax.set_xlabel('ESP Variance (kJ²/mol²)', fontweight='bold', labelpad=15)
ax.set_ylabel('HOMO-LUMO Gap (eV)', fontweight='bold', labelpad=15)
ax.set_zlabel('$\delta_{est}$ (eV)', fontweight='bold', labelpad=15)

# 调整3D视角设置，确保z轴标签可见
ax.view_init(elev=25, azim=45)

# 美化3D图
ax.grid(True, alpha=0.3)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# 添加颜色条 - 现在映射到delta_est
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.05)
cbar.set_label('$\delta_{est}$ (eV)', rotation=270, labelpad=20, fontweight='bold')

# 手动调整z轴标签位置（如果需要）
ax.zaxis.label.set_rotation(270)
ax.set_box_aspect(None, zoom=0.85)
plt.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.1)

plt.savefig('figs/figure_3d_main.png', dpi=600, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.savefig('figs/figure_3d_main.svg', bbox_inches='tight',transparent=True,)
plt.savefig('figs/figure_3d_main.pdf', bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()

# 创建XY投影图 - 显示ESP方差的极值分子
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
scatter = ax.scatter(x, y, c=z, cmap='viridis', s=15, alpha=0.7, edgecolors='none')

ax.set_xlabel('ESP Variance (kJ²/mol²)', fontweight='bold')
ax.set_ylabel('HOMO-LUMO Gap (eV)', fontweight='bold')

# 显示ESP方差的最小值和最大值
corner_extremes_xy = find_corner_points(x, y, smiles)
# extreme_indices_xy = [global_extremes['x_min'], global_extremes['y_min'],
extreme_indices_xy = [
                      corner_extremes_xy['left_bottom'], corner_extremes_xy['left_top'], 
                      corner_extremes_xy['right_bottom'], corner_extremes_xy['right_top']]

markers_xy = ['*', 'D', '^', 'h', '+', 'v']
colors_xy = ['red', 'blue','darkred', 'darkblue', 'darkgreen', 'darkorange']
# labels_xy = ['ESP方差最小值', 'ESP方差最大值']

used_positions_xy = []
used_bboxes_xy = []
for i, idx in enumerate(extreme_indices_xy):
    ax.scatter(x.iloc[idx], y.iloc[idx], c=colors_xy[i], s=120, marker=markers_xy[i], 
              edgecolors='black', linewidth=2, zorder=5)
    
    # 添加分子结构

    add_molecule_to_plot(ax, x.iloc[idx], y.iloc[idx], smiles.iloc[idx], used_positions_xy)

plt.tight_layout()
plt.savefig('figs/figure_xy_projection.png', dpi=600, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.savefig('figs/figure_xy_projection.svg', bbox_inches='tight',transparent=True,)
plt.savefig('figs/figure_xy_projection.pdf', bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()

# 创建XZ投影图 - 显示HOMO-LUMO Gap的极值分子
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
scatter = ax.scatter(x, z, c=z, cmap='viridis', s=15, alpha=0.7, edgecolors='none')

ax.set_xlabel('ESP Variance (kJ²/mol²)', fontweight='bold')
ax.set_ylabel('$\delta_{est}$ (eV)', fontweight='bold')

# 显示HOMO-LUMO Gap的最小值和最大值
corner_extremes_xz = find_corner_points(x, z, smiles)

# extreme_indices_xz = [global_extremes['x_max'], global_extremes['z_min'],
extreme_indices_xz = [
                      corner_extremes_xz['left_bottom'], corner_extremes_xz['left_top'], 
                      corner_extremes_xz['right_bottom'],corner_extremes_xz['right_top']]
markers_xz = ['*', 'D', '^', 'h', '+']
colors_xz = ['red', 'blue','darkred', 'darkblue', 'darkgreen', 'darkorange']


used_positions_xz = []
for i, idx in enumerate(extreme_indices_xz):
    ax.scatter(x.iloc[idx], z.iloc[idx], c=colors_xz[i], s=120, marker=markers_xz[i], 
              edgecolors='black', linewidth=2, zorder=5)
    
    # 添加分子结构
    add_molecule_to_plot(ax, x.iloc[idx], z.iloc[idx], smiles.iloc[idx], used_positions_xz)

plt.tight_layout()
plt.savefig('figs/figure_xz_projection.png', dpi=600, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.savefig('figs/figure_xz_projection.svg', bbox_inches='tight',transparent=True,)
plt.savefig('figs/figure_xz_projection.pdf', bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()

# 创建YZ投影图 - 显示Delta Est的极值分子
fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
scatter = ax.scatter(y, z, c=z, cmap='viridis', s=15, alpha=0.7, edgecolors='none')

ax.set_xlabel('HOMO-LUMO Gap (eV)', fontweight='bold')
ax.set_ylabel('$\delta_{est}$ (eV)', fontweight='bold')

# 显示Delta Est的最小值和最大值
corner_extremes_yz = find_corner_points(y, z, smiles)

# extreme_indices_yz = [global_extremes['y_max'], global_extremes['z_max'],
extreme_indices_yz = [
                      corner_extremes_yz['left_bottom'], corner_extremes_yz['left_top'], 
                      corner_extremes_yz['right_bottom'], corner_extremes_yz['right_top']]
markers_yz = ['*', 'D', '^', 'h', '+', 'v']
colors_yz = ['red', 'blue','darkred', 'darkblue', 'darkgreen', 'darkorange']
# labels_yz = ['Δ Est最小值', 'Δ Est最大值']

used_positions_yz = []
for i, idx in enumerate(extreme_indices_yz):
    ax.scatter(y.iloc[idx], z.iloc[idx], c=colors_yz[i], s=120, marker=markers_yz[i], 
              edgecolors='black', linewidth=2, zorder=5)
    
    # 添加分子结构
    add_molecule_to_plot(ax, y.iloc[idx], z.iloc[idx], smiles.iloc[idx], used_positions_yz)

plt.tight_layout()
plt.savefig('figs/figure_yz_projection.png', dpi=600, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.savefig('figs/figure_yz_projection.svg', bbox_inches='tight',transparent=True,)
plt.savefig('figs/figure_yz_projection.pdf', bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.close()

print("已生成四张独立图片:")
print("- figure_3d_main.png/pdf (3D主图)")
print("- figure_xy_projection.png/pdf (XY投影 - ESP方差极值分子)")
print("- figure_xz_projection.png/pdf (XZ投影 - HOMO-LUMO Gap极值分子)")
print("- figure_yz_projection.png/pdf (YZ投影 - Δ Est极值分子)")

# 打印详细信息
print("\n6个全局极值分子:")
extreme_names = {
    'x_min': 'ESP方差最小值',
    'x_max': 'ESP方差最大值',
    'y_min': 'HOMO-LUMO Gap最小值',
    'y_max': 'HOMO-LUMO Gap最大值',
    'z_min': 'Δ Est最小值',
    'z_max': 'Δ Est最大值'
}

for key, idx in global_extremes.items():
    print(f"  {extreme_names[key]}: {smiles.iloc[idx]}")

print("\nXY投影图 (ESP方差极值):")
for i, idx in enumerate(extreme_indices_xy):
    print(f"  ({colors_xy[i]}, {markers_xy[i]}): {smiles.iloc[idx]},{x.iloc[idx]},{y.iloc[idx]}")

print("\nXZ投影图 (HOMO-LUMO Gap极值):")
for i, idx in enumerate(extreme_indices_xz):
    print(f"  ({colors_xz[i]}, {markers_xz[i]}): {smiles.iloc[idx],{x.iloc[idx]},{z.iloc[idx]}}")

print("\nYZ投影图 (Δ Est极值):")
for i, idx in enumerate(extreme_indices_yz):
    print(f"  ({colors_yz[i]}, {markers_yz[i]}): {smiles.iloc[idx],{y.iloc[idx]},{z.iloc[idx]}}")


import rdkit