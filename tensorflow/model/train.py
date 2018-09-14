#-*-encoding=utf-8-*-
import tensorflow as tf
from base_model import lstm
from data.extractData import get_train_data
from modelSetting.model_setting import cfg

def train_lstm(input_size,rnn_unit,output_size,saver_dir,lr,max_to_keep,batch_size=80,time_step=15,train_end=5800):
    # X是训练数据中的feature Y是训练数据中的label
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size],name='input_x')
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    # 获取训练数据  batch_index:80的等差序列 train_x：[3785*15] train_y:[3785*15]  15:time_step值
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_end)
    # 创建预测值获取的计算流程
    pred,_=lstm(X,rnn_unit,input_size)
    # 创建损失函数的计算流程
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    # 定义优化函数（即训练使）
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    # 将变量保存
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=max_to_keep)
    with tf.Session() as sess:
        try:
            module_file = tf.train.latest_checkpoint('.')
            saver.restore(sess, module_file)
        except:
            sess.run(tf.global_variables_initializer())
        # 重复训练10000次
        for i in range(1000):
            # 按批次进行训练
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],
                                                            Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 200==0:
                print("保存模型：",saver.save(sess,saver_dir,global_step=i))

if __name__=="__main__":
    lr = cfg.LEARNING_RATE
    input_size = cfg.INPUT_SIZE
    output_size = cfg.OUTPUT_SIZE
    batch_size = cfg.BATCH_SIZE
    time_step = cfg.TIME_STEP
    train_end = cfg.TRAIN_END
    max_to_keep = cfg.MAX_TO_KEEP
    saver_dir = cfg.SAVER_DIR
    rnn_unit = cfg.RNN_UNIT
    train_lstm(input_size,rnn_unit,output_size,saver_dir,lr,max_to_keep,batch_size,time_step,train_end)
