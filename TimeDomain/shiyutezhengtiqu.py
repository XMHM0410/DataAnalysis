from tiqutezheng import split_sample_tolist, get_feature
import pandas as pd
import numpy as np






if __name__ =='__main__':

    cols = ['zd_x', 'zd_y','zd_z','dl_U','dl_W','dl_V']
    nrows = None


    #创建特征集
    df = pd.DataFrame()
    for i in range(1,81,1):
        
        filepath = str(i*100)+'r.csv'

    #加载原始数据
        data = pd.read_csv(filepath)
        data = data.loc[0:10000,cols]

        
        #切分若干个样本
        split_samples = split_sample_tolist(data, 1000)

        
        #提取每个样本的PFs_RMS特征
        for sample in split_samples:
            shiyutezheng_dict = {'rpm':i*100}

            for col in cols:
                shiyutezheng_dict.update(get_feature(sample[col].to_list(),col))
                
            df = df.append(shiyutezheng_dict,ignore_index=True)

    #保存特征集
    df.to_csv('./tezheng_data/yshiyutezheng_data.csv')