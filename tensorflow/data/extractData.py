#-*-encoding=utf-8-*-

import pandas as pd
import numpy as np
from dataSetting.dataSetting import cfg

def loadData():
    data_dir = cfg.DATA_PATH #要更改只需要在配置文件中设置即可
    data = pd.read_csv(data_dir)
    return data.iloc[:,2:10]

def get_train_data(batch_size=60,time_step=15,train_nums=5800):
    data = loadData()
    batch_index=[]
    #获取训练数据
    data_train=data[:train_nums]
    # 标准化
    normalized_train_data=(data_train-data_train.mean(axis=0))/data_train.std(axis=0)
    # 由于要使用时序网络，因此数据要弄成时序形式的，也就是一条数据包含多个时间点的元数据
    train_x,train_y=[],[]
    # 此时normalized_train_data的shape是n*8
    for i in range(len(normalized_train_data)-time_step):       # i = 1~5785
        # batch_index,为等差序列，方便数据批读取，一批数据中，有batch_size条数据，每条数据有time_step个元数据,
        #每一个元数据就是一行数据
        if i % batch_size==0:
            batch_index.append(i)
        x=normalized_train_data.iloc[i:i+time_step,:7]                # x:shape 15*7
        y=normalized_train_data.iloc[i:i+time_step,7]      # y:shape 15*1
        train_x.append(x.values.tolist())
        train_y.append(y.values.reshape(time_step,1).tolist())
    batch_index.append((len(normalized_train_data)-time_step))  # batch_index 收尾
    return batch_index,train_x,train_y

def get_data(batch_size=60,time_step=15,test_begin=5800):
    data = loadData()
    batch_index=[]
    test=data[test_begin:] # 截取测试数据
    normalized_test_data=(test-test.mean(axis=0))/test.std(axis=0)   # 标准化
    # 有size个sample
    size=(len(normalized_test_data)+time_step-1)//time_step
    test_x = []
    test_y = normalized_test_data.iloc[:,7] #只需要取出最后一位即可，因为本次是预测网络，所以不需要把y也输进网络
    for i in range(size-1):
        #每一个元数据就是一行数据
        if i % batch_size==0:
            batch_index.append(i)
        x=normalized_test_data.iloc[i*time_step:(i+1)*time_step,:7]   # x shape time_step*7
        test_x.append(x.values.tolist())#toList()一定要加括号，否则会出现内建函数的错误
    batch_index.append((len(normalized_test_data)-time_step))  # batch_index 收尾,最后一批数据可能有重复
    return batch_index,test_x,test_y

if __name__ == "__main__":
    # get_train_data()
    get_data()
