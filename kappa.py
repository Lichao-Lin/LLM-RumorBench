import pandas as pd
from sklearn.metrics import cohen_kappa_score

def simple_kappa_test(file_path):
 
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 提取第1-2列数据（索引0和1）
    col1 = df.iloc[:, 0].dropna().astype(int)
    col2 = df.iloc[:, 1].dropna().astype(int)
    
    # 提取第3-4列数据（索引2和3）
    col3 = df.iloc[:, 2].dropna().astype(int)
    col4 = df.iloc[:, 3].dropna().astype(int)
    
    # 确保数据长度一致
    min_len_12 = min(len(col1), len(col2))
    col1 = col1.iloc[:min_len_12]
    col2 = col2.iloc[:min_len_12]
    
    min_len_34 = min(len(col3), len(col4))
    col3 = col3.iloc[:min_len_34]
    col4 = col4.iloc[:min_len_34]
    
    # 第1-2列：普通Kappa检验
    kappa_12 = cohen_kappa_score(col1, col2)

    #3-4，普通kappa
    kappa_34 = cohen_kappa_score(col3, col4)
    
    # 第3-4列：加权Kappa检验（线性权重）
    weighted_kappa_34 = cohen_kappa_score(col3, col4, weights='linear')
    
    # 输出结果
    print(f"第1-2列是否 Cohen's Kappa: {kappa_12:.4f}")
    print(f"第3-4列频率 Cohen's Kappa: {kappa_34:.4f}")
    print(f"第3-4列频率 Weighted Kappa (linear): {weighted_kappa_34:.4f}")
    
    return kappa_12,kappa_34,weighted_kappa_34

# 
if __name__ == "__main__":
   
    excel_file = "C:\\Users\\megamind\\Desktop\\a.xlsx"
    
    try:
        kappa_result, weighted_kappa_result = simple_kappa_test(excel_file)
    except Exception as e:
        print(f" {e}")
        