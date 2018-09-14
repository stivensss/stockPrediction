#-*-encoding=utf-8-*-

import tensorflow as tf
from data.extractData import get_data
import matplotlib.pyplot as plt
from modelSetting.model_setting import  cfg

def prediction(time_step,saver_dir):
    batch_index,test_x,test_y=get_data(time_step)
    with tf.Session() as sess:
        modelDir = saver_dir
        saver = tf.train.import_meta_graph(modelDir+'.meta') #meta保存的是神经网络模型图
        # saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir='ckpt')) #加载最后一次模型
        saver.restore(sess,modelDir) #这种方式可以使用任意一个保存的模型进行预测，这其中包括了各个参数
        graph = tf.get_default_graph()#获取神经网络图
        X = graph.get_operation_by_name('input_x').outputs[0]#获取输入参数x，根据命名获取
        pred = graph.get_operation_by_name('pred').outputs[0]#获取预测值，根据命名获取
        test_predict = []
        for step in range(len(batch_index)-1):
            prob=sess.run(pred,feed_dict={X:test_x[batch_index[step]:batch_index[step+1]]})
            predict=prob.reshape((-1))
            test_predict.extend(predict)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

if __name__ == "__main__":
    lr = cfg.LEARNING_RATE
    output_size = cfg.OUTPUT_SIZE
    batch_size = cfg.BATCH_SIZE
    time_step = cfg.TIME_STEP
    train_end = cfg.TRAIN_END
    max_to_keep = cfg.MAX_TO_KEEP
    predict_dir = cfg.PREDICT_DIR
    prediction(time_step,predict_dir)