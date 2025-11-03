import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

# mode = 'delta_est'
# mode = 'esp2'
mode = 'esp2qm'

aaa = pd.read_csv(f'trans/{mode}.txt')
# aaa = pd.read_csv(f'data/trans_{mode}.csv')
# aaa = pd.read_csv(f'figs/{mode}.txt')

trues = aaa['True'].to_list()
preds = aaa['Pred'].to_list()

if mode == 'homo_lumo_gap' or mode == 'trans_homo_lumo_gap'or mode == 'gap2rtp' or mode == 'gap2esp':
    title = 'HOMO/LUMO Gap'
    unit = 'eV'
    trues = (27.2114 * aaa['True']).to_list()
    preds = (27.2114 * aaa['Pred']).to_list()
    color = '#52b788'
    edgecolors='#2d6a4f'

elif mode == 'delta_est'or mode == 'rtp2esp' or mode == 'rtp2qm':
    title = '$\delta_{est}$'
    unit = 'eV'
    color = '#8ecae6'
    edgecolors='#0077b6'

elif mode == 'esp':
    title = '$ESP variance$'
    unit = '$a.u.^2$'
elif mode == 'esp2' or mode == 'esp2qm' or mode == 'esp2rtp':
    title = '$ESP variance$'
    unit = '$kJ^2/mol^2$'
    color = '#d90429'
    edgecolors='#800e13'

plt.figure(figsize=(8, 8))
plt.scatter(trues, preds, color=color, s=10, alpha=0.6, edgecolors=edgecolors)
min_val, max_val = np.min([np.min(trues), np.min(preds)]), np.max([np.max(trues), np.max(preds)])
plt.plot([min_val, max_val], [min_val, max_val], 'k--')

# 设置相同的坐标轴范围和刻度
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# 确保x和y轴有相同的刻度
from matplotlib.ticker import MaxNLocator
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置最多6个主要刻度
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置最多6个主要刻度

# 或者手动设置相同的刻度位置
# ticks = np.linspace(min_val, max_val, 6)
# plt.xticks(ticks)
# plt.yticks(ticks)

r2 = r2_score(trues, preds)
rmse = np.sqrt(mean_squared_error(trues, preds))
mae = mean_absolute_error(trues, preds)

plt.text(0.02, 0.98,
         f'$r^2 = {r2:.4f}$\n$RMSE = {rmse:.4f}$ {unit}\n$MAE = {mae:.4f}$ {unit}', 
         fontsize=18, style='italic',
         ha='left', va='top',
         transform=plt.gca().transAxes)

plt.xlabel(f'Calculated {title} ({unit})', fontsize=22, fontstyle='italic')
plt.ylabel(f'Predicted {title} ({unit})', fontsize=22, fontstyle='italic')
# plt.title(f'{title}', fontsize=24, fontstyle='italic')

# 设置刻度标签大小
plt.tick_params(axis='both', which='major', labelsize=20)

# 确保标题显示完全，调整图片边距
plt.tight_layout()

# 保存时设置边距参数
plt.savefig(f'{mode}.png', dpi=300)
plt.savefig(f'{mode}.svg', bbox_inches='tight', pad_inches=0.1, transparent=True)
plt.show()