
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
import argparse
import os

def selection_feature(data,method):
    x,y=data.iloc[:,1:],data.iloc[:,0]
    
    if method =='RF':
        svm_clf=RandomForestClassifier(random_state=0)
        svm_clf.fit(x,y)
        feature_import=svm_clf.feature_importances_
       
    if method =='XGBoost':
        xgb_clf=XGBClassifier()
        xgb_clf.fit(x,y)
        feature_import=xgb_clf.feature_importances_ 
       
    if method =='CATboost':
        cat_clf=CatBoostClassifier()
        cat_clf.fit(x,y)
        feature_import=cat_clf.feature_importances_ 
              
    forest_importance=pd.DataFrame({'feature_name':x.columns,'scores':feature_import})
    forest_importance_sort=forest_importance.sort_values(by='scores',ascending=False)
    outfile= open(method+'.csv','w',newline='',encoding='utf-8')
    print('index',end='\t',file=outfile)
    print("origin_index",end='\t',file=outfile)
    print("features",end='\t',file=outfile)
    print("features_score",file=outfile)
    for i in range(len(forest_importance_sort)):
        print(i+1,end='\t',file=outfile)
        print(forest_importance_sort.index[i]+1,end='\t',file=outfile)
        print(forest_importance_sort.iloc[i,:][0],end='\t',file=outfile)
        print(forest_importance_sort.iloc[i,:][1],file=outfile)
    
#    forest_importance_sort.to_csv('out/{}.csv'.format(method))  
    return
    
def main(args):
    # 读取数据
    pwd=os.path.abspath(__file__)
    df =pd.read_csv(os.path.dirname(pwd)+'/'+args.input_data,dtype='float')
    data=df.fillna(0)
    methods=['RF','XGBoost','CATboost']
    if args.method not in methods:
             print('\nyour method is not available')
             print('please select model in：'+methods[0]+','+methods[1]+','+methods[2])
    else:
         selection_feature(data,args.method)
      
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM_RF_under/up_sample')
    parser.add_argument('--input_data','-i',type=str,
                        help='The path of dataset.' ) 
    parser.add_argument("--method", "-name", type=str,
                        help="The feature selection model")
    args = parser.parse_args()
    main(args)
