# -*- coding: utf-8 -*-
"""
A random forest classifier based on scikit-learn with SMOTE apply it to classify bio datasets.
Date: 2020-03-01
"""
import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'

if __name__ == "__main__":
    start_time = time.time()
    # region Parameter
    parser = ArgumentParser(
        description="A random forest classifier based on scikit-learn with SMOTE apply it to classify bio datasets.",
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--ntrees", type=int,
                        help="The number of trees in the forest.", default=100)
    parser.add_argument("-a", "--attributes", type=int,
                        help="Number of attributes.", default=1)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=100)
    parser.add_argument("-k", "--features", type=int,
                        help="Number of folds. Must be at least 20.", default=20)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=False)

    args = parser.parse_args()
    # logdir_base = os.getcwd()  # 获取当前目录
    # endregion

    # 导入相关库
    import numpy as np
    import pandas as pd

    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_rows', None)  # 显示所有行
    # np.set_printoptions(threshold=np.inf)  # 显示所有数据

    # 读取数据
    df = pd.read_csv(args.datapath)
    file_name = os.path.basename(args.datapath).split('.')[0]

    # print(df.columns[1:])
    # 设定分类信息和特征矩阵
    X = df.iloc[:, args.attributes:].values
    y = df.iloc[:, :args.attributes].values.flatten()

    # 不同 Class 统计 (根据 Target 列)
    # print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    num_categories = len(np.unique(y, axis=0))

    # 数据标准化
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # 特征 选择
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

    selector = SelectKBest(chi2, k=args.features)
    #selector = SelectKBest(mutual_info_classif, k=args.features)
    # print(X.shape)
    X_select = selector.fit_transform(X_scaled, y)
    # print(X_select.shape)

    # 获取选择的特征索引
    selector_fit = selector.fit(X_scaled, y)
    selected_features_indices = selector_fit.get_support(indices=True)
    selected_feature_scores = selector_fit.scores_[selected_features_indices]
    # 使用索引获取原始数据集中选择的特征名
    selected_feature_names = df.columns[args.attributes:][selected_features_indices]
    # 创建 DataFrame 并保存为 CSV 文件
    result_df = pd.DataFrame({
        'origin_index': selected_features_indices + args.attributes,
        'features': selected_feature_names,
        'features score': selected_feature_scores
    })

    # 按照特征得分降序排序
    result_df = result_df.sort_values('features score', ascending=False)

    result_df.index = list(range(1, len(selected_feature_names) + 1))
    result_df.index.name = 'index'

    print(result_df)
    result_df.to_csv(f'k_{args.features}_{file_name}.csv', index=True, sep="\t")

    end_time = time.time()  # 程序结束时间
    # print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(((end_time - start_time) / 60),
    #                                                                (end_time - start_time)))
