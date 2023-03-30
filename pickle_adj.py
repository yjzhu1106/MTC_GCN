import numpy as np
import pandas as pd
import os
import pickle

from  scipy.stats import spearmanr
def get_va_va_spearmanr(df):
    # 选择所有带有愤怒标签的样本
    # 基于条件创建布尔索引

    valence = df['valence'].values
    arousal = df['arousal'].values

    # 计算愤怒和valence之间的斯皮尔曼相关系数
    corr, pval = spearmanr(valence, arousal)
    corr2, pval2 = spearmanr(arousal, valence)

    # print(expression_value, "和valence之间的斯皮尔曼相关系数：", round(corr, 6))
    # print(expression_value, "和arousal之间的斯皮尔曼相关系数：：", round(corr2, 6))
    # print("valence和",expression_value, "之间的斯皮尔曼相关系数：", round(corr3, 6))
    # print("arousal和",expression_value, "之间的斯皮尔曼相关系数：：", round(corr4, 6))
    return round(corr, 5),round(corr2, 5)


def get_exp_va_spearmanr(df, expression_value):
    # 选择所有带有愤怒标签的样本
    # 基于条件创建布尔索引
    bool_idx = df["expression"] == expression_value

    # 设置新列的值
    df.loc[bool_idx, "expression_value"] = 1
    df.loc[~bool_idx, "expression_value"] = 0

    # 提取出每个样本的愤怒和valence标签
    expression = df['expression_value'].values
    valence = df['valence'].values
    arousal = df['arousal'].values

    # 计算愤怒和valence之间的斯皮尔曼相关系数
    corr, pval = spearmanr(expression, valence)
    corr2, pval2 = spearmanr(expression, arousal)
    corr3, pval = spearmanr(valence, expression)
    corr4, pval2 = spearmanr(arousal, expression)

    # print(expression_value, "和valence之间的斯皮尔曼相关系数：", round(corr, 6))
    # print(expression_value, "和arousal之间的斯皮尔曼相关系数：：", round(corr2, 6))
    # print("valence和",expression_value, "之间的斯皮尔曼相关系数：", round(corr3, 6))
    # print("arousal和",expression_value, "之间的斯皮尔曼相关系数：：", round(corr4, 6))
    return round(corr, 5),round(corr2, 5),round(corr3, 5),round(corr4, 5)



def get_exp_exp_spearmanr(df, i, j):
    # 选择所有带有愤怒标签的样本
    # 基于条件创建布尔索引
    bool_idx_i = df["expression"] == i
    bool_idx_j = df["expression"] == j

    # 设置新列的值
    df.loc[bool_idx_i, "i"] = 1
    df.loc[~bool_idx_i, "i"] = 0

    df.loc[bool_idx_j, "j"] = 1
    df.loc[~bool_idx_j, "j"] = 0

    # 提取出每个样本的愤怒和valence标签
    expression_i = df['i'].values
    expression_j = df['j'].values

    # 计算愤怒和valence之间的斯皮尔曼相关系数
    corr, pval = spearmanr(expression_i, expression_j)


    # print(expression_i, "和", expression_j, "之间的斯皮尔曼相关系数：", round(corr, 6))
    # print(expression_j, "和", expression_i, "之间的斯皮尔曼相关系数：：", round(corr2, 6))

    return round(corr, 5)



if __name__ == '__main__':
    affenet_spearman = pickle.load(open('/root/autodl-tmp/AffectNet/data/spearman_affectnet.pkl', 'rb'))
    print(affenet_spearman)

    save_path = '/root/autodl-tmp/RAF-DB/data/spearman_raf_db.pkl'
    csv_dir = '/root/autodl-tmp/RAF-DB/data'

    df_train = pd.read_csv(os.path.join(csv_dir, 'training.csv'))

    df_train = df_train[df_train['expression'] < 7]

    df = df_train
    corr_result = []
    for i in range(7):
        corr_i = []
        for j in range(7):
            corr = get_exp_exp_spearmanr(df, i, j)
            corr_i = np.append(corr_i, corr)
        corr_va, corr2_va, corr3_va, corr4_va = get_exp_va_spearmanr(df, i)
        corr_i = np.append(corr_i, corr_va)
        corr_i = np.append(corr_i, corr2_va)

        corr_result.append(corr_i)

    corr_i = []
    for i in range(8):
        if i >= 7:
            corr_va_va, corr2_va_va = get_va_va_spearmanr(df)
            corr_i = np.append(corr_i, 1.0)
            corr_i = np.append(corr_i, corr_va_va)
            continue
        corr_va, corr2_va, corr3_va, corr4_va = get_exp_va_spearmanr(df, i)
        corr_i = np.append(corr_i, corr_va)
    corr_result.append(corr_i)

    corr_i = []
    for i in range(8):
        if i >= 7:
            corr_va_va, corr2_va_va = get_va_va_spearmanr(df)
            corr_i = np.append(corr_i, corr2_va_va)
            corr_i = np.append(corr_i, 1.0)
            continue
        corr_va, corr2_va, corr3_va, corr4_va = get_exp_va_spearmanr(df, i)
        corr_i = np.append(corr_i, corr2_va)
    corr_result.append(corr_i)


    corr_result = np.array(corr_result)
    df_result = pd.DataFrame(corr_result)
    df_result = df_result.rename(columns={
                            0: 'Neutral',
                            1: 'Happy',
                            2: 'Sad',
                            3: 'Surprise',
                            4: 'Fear',
                            5: 'Disgust',
                            6: 'Anger',
                            7: 'Valance',
                            8: 'Arousal'
                            })
    df_result.index = ['Neutral', 'Happy', 'Sad','Surprise', 'Fear', 'Disgust','Anger', 'Valance', 'Arousal']

    with open(save_path, 'wb') as f:
        pickle.dump(df_result, f)

    print(df_result)

    # corr=(i,arousal),corr2=(i,valance),corr3=(arousal, i),corr4=(valnace, i) 错误的
    # corr=(i,valance),corr2=(i,arousal),corr3=(valance,i),corr4=(arousal,i) 正确的
    # corr, corr2, corr3, corr4 = get_exp_va_spearmanr(df, 0)
    # print("(i,arousal):", corr ,"; (i,valance)", corr2,"; (arousal, i)",  corr3, "; (valnace, i)", corr4) # 错误的
    # print("(i,valance):", corr, "; (i,arousal)", corr2, "; (valance, i)", corr3, "; (arousal, i)", corr4)  # 正确的的

    # corr, corr2 = get_exp_exp_spearmanr(df, 0, 1)
    # print("(i, j): ", corr, "; (j, i): ", corr2)












