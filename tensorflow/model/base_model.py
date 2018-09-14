# encoding: utf-8

from __future__ import print_function
import tensorflow as tf

# 根据股票历史数据中的最低价、最高价、开盘价、收盘价、交易量、交易额、跌涨幅等因素，对下一日股票最高价进行预测。
def lstm(X,rnn_unit,input_size):
    # X:shape=[None,time_step,input_size]
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in = tf.Variable(tf.random_normal([input_size,rnn_unit]))
    b_in = tf.Variable(tf.constant(0.1,shape=[rnn_unit,]))
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入 系统的输入单元数为7，所以将输入数据shape为n行7列
    input_rnn=tf.matmul(input,w_in)+b_in    # 输入数据与输入权重相乘，得到输入数据对隐含层的影响
    # 将tensor转成3维，作为lstm cell的输入
    # 隐含层的cell接收的数据是3维的，即将n*10的数据shape为n*15*10的数据
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    #采用GRU模型
    cell2 = tf.nn.rnn_cell.GRUCell(rnn_unit)
    cell2_init_state = cell2.zero_state(batch_size,dtype=tf.float32)
    # output_rnn是记录GRU每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn,final_states=tf.nn.dynamic_rnn(cell2, input_rnn,initial_state=cell2_init_state, dtype=tf.float32)
    # 将输出数据shape为n*10格式
    output=tf.reshape(output_rnn,[-1,rnn_unit])
    w_out = tf.Variable(tf.random_normal([rnn_unit,1]))
    b_out = tf.Variable(tf.constant(0.1,shape=[1,]))
    # cell输出经过与输出权重矩阵相乘并加入偏置后，得到最终输出
    pred=tf.add(tf.matmul(output,w_out),b_out,name="pred")

    return pred,final_states
