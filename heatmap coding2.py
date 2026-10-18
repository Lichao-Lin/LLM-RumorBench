import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

#定义列
CUSTOM_METHOD_COLS = [
    "correct answer",
    "Neither",
    "web search",
    "deep think",
    "Both"
]

def main():
    parser = argparse.ArgumentParser(description="Numeric correlation, not agreement; numeric or ordinal codes only")
    parser.add_argument("input")
    parser.add_argument("--output", default="outputs/correlation.png")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="spearman")
    args = parser.parse_args()
    try:
        # 读取 Excel
        df = pd.read_excel(args.input)

        # 提取列
        df_subset = df[CUSTOM_METHOD_COLS].apply(pd.to_numeric, errors="raise").dropna()
        if len(df_subset) < 2:
            raise ValueError("At least two common complete rows are required")

        # 所有条件使用共同完整样本
        corr_matrix = df_subset.corr(method=args.method)

        # 画图
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="vlag",
            linewidths=0.5,
            linecolor="#444444",
            square=True,
            vmin=-1,
            vmax=1,
            cbar_kws={"label": args.method}
        )

        # 添加标题与标签
        plt.title(f"{args.method.title()} correlation (n={len(df_subset)})", fontsize=14, pad=20)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存
        output_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        raise SystemExit(f"error: {e}")

if __name__ == "__main__":
    main()
