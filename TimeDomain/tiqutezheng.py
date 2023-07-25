from distutils.log import debug
import imp
import numpy as np
import pandas as pd
from PyLMD import LMD
import matplotlib.pyplot as plt
import math




def split_sample_tolist(df, sample_length=1000):
    """ 将信号截成若干个样本 """
    result = []
    for i in range(0,len(df),sample_length):
        if i+ sample_length-1 <len(df):
           result.append(df.loc[i:(i+sample_length-1),:])
    return result




def get_PFs(df,col):
    """ df分解成活干个PF分量 return PFs, res """
    lmd = LMD()
    y = np.array(df[col])
    print(y) if debug == True  else None
    PFs, res = lmd.lmd(y)
    return PFs, res

def get_PFs_RMS(PFs, pf_num,col):
    """ 获取PFs的RMS值，return -> dict """
    RMS_dict = {}
    for i in range(pf_num):
        key = col+'_pf'+str(i+1)+'_rms'
        RMS_dict[key] = round(math.sqrt(sum([x ** 2 for x in PFs[i]]) / len(PFs[i])),5)
    return RMS_dict


def normalize(df:pd.DataFrame):
    """ 标准归一化 """
    data_mean = df.mean(axis=0)
    data_std = df.std(axis=0)
    return (df - data_mean) / data_std


def get_feature(data_list,col):
    """均方根值 反映的是有效值而不是平均值 """
    X_rms = math.sqrt(sum([x ** 2 for x in data_list]) / len(data_list))
    """峰峰值"""
    X_p_p = max(data_list) - min(data_list)
    """峰值"""
    X_p = max([abs(x) for x in data_list])
    """平均幅值"""
    X_ma = sum([abs(x) for x in data_list]) / len(data_list)
    """方根幅值"""
    X_r = pow(sum([math.sqrt(abs(x)) for x in data_list]) / len(data_list), 2)
    """峰值因子"""
    C_f = X_p / X_rms
    """波形因子"""
    C_s = X_rms / X_ma
    """脉冲因子"""
    C_if = X_p / X_ma
    """裕度因子"""
    C_mf = X_p / X_r
    """峭度因子"""
    C_kf = (sum([x ** 4 for x in data_list]) / len(data_list)) / pow(X_rms, 4)

    feature = {col+'X_rms':round(X_rms, 3), col+'X_p_p': round(X_p_p, 3), col+'C_f':round(C_f, 3), col+'C_s':round(C_s, 3),
            col+'C_if':round(C_if, 3), col+'C_mf':round(C_mf, 3), col+'C_kf':round(C_kf, 3)}

    return feature


if __name__ =='__main__':
    #原代码
    # debug = False
    # pf_num = 3
    # cols = ['zd_x', 'zd_y','zd_z',]
    # nrows = None
    # #创建特征集
    # df = pd.DataFrame()
    # for i in range(1,81,1):        
    #     filepath = str(i*100)+'r.csv'
    # #加载原始数据
    #     data = pd.read_csv(filepath)
    #     data = data.loc[0:20000,cols]
    #     # print(data) if debug == True  else None  
    #     #切分若干个样本
    #     split_samples = split_sample_tolist(data, 1000)
    #     # #创建特征集
    #     # df = pd.DataFrame()
    #     #提取每个样本的PFs_RMS特征
    #     for sample in split_samples:
    #         pf_rms_dict = {'rpm':i*100}
    #         for col in cols:
    #             PFs,res = get_PFs(sample, col)
    #             pf_rms_dict.update(get_PFs_RMS(PFs,pf_num,col))
    #         df = df.append(pf_rms_dict,ignore_index=True)
    #     # print(df) if debug == True  else None
    # #保存特征集
    # df.to_csv('./tezheng_data/PFs_rms.csv')

    #声发射代码20220306
    debug=False
    pf_num = 1
    cols=['col_1']
    df = pd.DataFrame()
    nrows = None
    data = pd.read_csv('100.csv')
    data = data.loc[0:10000,cols]
    split_samples = split_sample_tolist(data,1000)
    for sample in split_samples:
        pf_rms_dict = {'rpm':100}
        for col in cols:
            PFs,res = get_PFs(sample, col)
            pf_rms_dict.update(get_PFs_RMS(PFs,pf_num,col))
        df = df.append(pf_rms_dict,ignore_index=True)
    df.to_csv('./tezheng_data/PFs_rms.csv')
    