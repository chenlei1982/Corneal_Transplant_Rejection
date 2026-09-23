import numpy as np
from sklearn import manifold, datasets
from sklearn.utils import check_random_state
import pandas as pd
import argparse
import os

if not os.path.exists('out/'):
    os.makedirs('out/')

def selection_feature(data,n,c):
    x,y=data.iloc[:,1:],data.iloc[:,0].astype(int)
    lle_sr,lle_er=manifold.locally_linear_embedding(x,n_neighbors=n, n_components=c)
    new_feature=pd.DataFrame(lle_sr,columns=['new_features'+str(i) for i in range(0,c)],index=y)
    new_feature.to_csv('out/{}.csv'.format('LLE'))  
    return
    
def main(args):
    # 读取数据
    pwd=os.path.abspath(__file__)
    df =pd.read_csv(os.path.dirname(pwd)+'/'+args.input_data,dtype='float')
    data=df.fillna(0)
    selection_feature(data,args.n_neighbors,args.n_components)
      
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLE')
    parser.add_argument('--input_data','-i',type=str,
                        help='The path of dataset.' ) 
    parser.add_argument("--n_neighbors", "-n", type=int,
                        help="The number of neighbors")
    parser.add_argument("--n_components", "-c", type=int,
                        help="The number of components")
                        
    args = parser.parse_args()
    main(args)
