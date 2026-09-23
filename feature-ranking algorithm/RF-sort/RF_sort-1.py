# -*- coding: utf-8 -*-
"""
LightGBM_sort
"""
# import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


__author__ = 'LU'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="RF_sort.py",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    # 用pandas读取
    dataset = pd.read_csv(args.datapath)

    # 分离data(X)和label(Y)
    col = dataset.columns.values.tolist()  # 取第一行
    col1 = col[1:]  # 取特征
    X = np.array(dataset[col1])  # 取数据
    Y = preprocessing.LabelEncoder().fit_transform(dataset['label'])  # 标签标准化
    feature = list(dataset.columns.values)  # 同col
    names_ = feature[1:]  # 同col1

    model = RandomForestClassifier(random_state=args.randomseed)
    model.fit(X, Y)
    rows = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names_), reverse=True) # feature_importances等同于随机森林中的importance
    res_list = [x[1] for x in rows]
    score = [x[0] for x in rows]
    sums = []
    for a in res_list:
        sums.append(str(names_.index(str(a)) + 1))
    for a in range(1, len(col1)+1):
        print(str(a) + " " + str(sums[a - 1]) + " " + str(res_list[a - 1]) + " " + str(score[a - 1]))

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))

