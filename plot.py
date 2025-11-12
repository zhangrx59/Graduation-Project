import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
diseases = ['基底细胞癌', '皮肤纤维瘤', '血管病变', '光化性角化病',
           '黑色素瘤', '良性角化病', '色素痣']
percentages = [0.05, 1.15, 1.42, 3.27, 11.13, 10.99, 67.05]

# 颜色设置（根据病变性质）
colors = ['#FF4757', '#2ED573', '#2ED573', '#FFA502', '#FF4757', '#2ED573', '#1E90FF']

# 创建图形
plt.figure(figsize=(12, 8))
bars = plt.bar(diseases, percentages, color=colors, alpha=0.8)

# 设置标题和标签
plt.title('皮肤病变类型分布', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('百分比 (%)', fontsize=12)
plt.xlabel('病变类型', fontsize=12)

# 旋转x轴标签
plt.xticks(rotation=45, ha='right')

# 在柱子上添加数值标签
for bar, percentage in zip(bars, percentages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{percentage}%', ha='center', va='bottom', fontsize=10)

# 添加网格和调整y轴范围
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, max(percentages) + 5)

# 添加图例说明颜色含义
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF4757', alpha=0.8, label='恶性病变'),
    Patch(facecolor='#FFA502', alpha=0.8, label='癌前病变'),
    Patch(facecolor='#2ED573', alpha=0.8, label='良性病变'),
    Patch(facecolor='#1E90FF', alpha=0.8, label='常见良性病变')
]
plt.legend(handles=legend_elements, loc='upper right')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('skin_lesion_simple.png', dpi=300, bbox_inches='tight')
plt.show()