import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import argparse

def main(args):
    #写出结果的文件路径
    outfile = open(args.output_data,'w',newline='',encoding='utf-8')
    # 读取数据
    df = pd.read_csv(args.input_data)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values - 1
    f_names = df.columns[1:].values
    # print("index:",list(range(1,len(f_names))),file=outfile)
    print("index",end='\t',file=outfile)
    print("origin_index",end='\t',file=outfile)
    print("features",end='\t',file=outfile)
    print("features score",file=outfile)
    # print(list(range(1,len(f_names)+1)),file=outfile)

    # pre_progressing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    lasso_ = Lasso(alpha=args.alpha,random_state=args.randomseed,max_iter=args.maxiter).fit(X,y)
    coef = (lasso_.coef_*100).tolist()
    out_dict = {}
    for i in range(len(coef)):
        out_dict[str(i+1),f_names[i]] = coef[i]
    
    new_list = sorted(out_dict.items(), key=lambda item:item[1], reverse=True)
    # origin_index_list = []
    # features_name = []
    # features_score = []
    for i in range(len(new_list)):
        print(i+1,end='\t',file=outfile)
        print(new_list[i][0][0],end='\t',file=outfile)
        print(new_list[i][0][1],end='\t',file=outfile)
        print(new_list[i][1],file=outfile)
        # origin_index_list.append(new_list[i][0][0])
        # features_name.append(new_list[i][0][1])
        # features_score.append(new_list[i][1])
    # print("origin_index:",end=',',file=outfile)
    # print(origin_index_list,file=outfile)
    # print("features:",end=',',file=outfile)
    # print(features_name,file=outfile)
    # print("features score:",end=',',file=outfile)
    # print(features_score,file=outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature_selection_by_Lasso')
    parser.add_argument('--input_data','-i',type=str,
                        help='The path of dataset.' ) 
    parser.add_argument('--output_data','-o',type=str,
                        help='The path of output result.' ) 
    parser.add_argument('--alpha','-a',type=float,
                        help='Lasso`s Regularization coefficient',default=0.01 ) 
    parser.add_argument("--maxiter", "-m", type=int,
                        help="The max_iter", default=1000)
    parser.add_argument("--randomseed", "-r", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=2021)
    args = parser.parse_args()
    main(args)