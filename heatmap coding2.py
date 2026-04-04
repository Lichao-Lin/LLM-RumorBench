import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# 文件路径
EXCEL_PATH = r'C:\Users\asus\Desktop\Result List.xlsx'

#定义列
CUSTOM_METHOD_COLS = [
    "correct answer",
    "Neither",
    "web search",
    "deep think",
    "Both"
]

# 定义颜色映射
CUSTOM_COLORS = [
   "#FFEDA0",  
    "#FED976",  # 橙黄
    "#FEB24C",  # 橙
    "#FD8D3C",  # 橙红
    "#FC4E2A",  # 红
    "#E31A1C",  # 深红
    "#B10026"   # 暗红 (最大值)
]

def main():
    try:
        # 读取 Excel
        df = pd.read_excel(EXCEL_PATH)

        # 提取列
        df_subset = df[CUSTOM_METHOD_COLS]

        # Pearson 相关性
        corr_matrix = df_subset.corr(method='pearson')

        # 画图
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_orange_red", CUSTOM_COLORS, N=256
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=custom_cmap,
            linewidths=0.5,
            linecolor="#444444",
            square=True,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Pearson "}
        )

        # 添加标题与标签
        plt.title("consistent heat map", fontsize=14, pad=20)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存
        output_path = os.path.join(os.getcwd(), 'custom_method_consistency_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"worn: {e}")

if __name__ == "__main__":
    main()
