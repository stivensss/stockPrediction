#-*-encoding=utf-8-*-

from easydict import EasyDict as edict

_D = edict()
cfg = _D
#模型学习率
_D.LEARNING_RATE = 0.0001
#模型输入输出
_D.INPUT_SIZE = 7
_D.OUTPUT_SIZE = 1
#训练
_D.BATCH_SIZE = 80 #批数据
_D.TIME_STEP = 15 #时间步长
_D.TRAIN_END = 5800 #数据量
_D.STEP = 1000  #训练迭代次数
#模型
_D.MAX_TO_KEEP = 15
_D.RNN_UNIT = 10
_D.SAVER_DIR = './model_checkPoint/stock2.model'
_D.PREDICT_DIR = './model_checkPoint/stock2.model-800'
