#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 小杜的nlp乐园 欢迎关注
#Author 小杜好好干
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Dropout, Flatten, MaxPool2D, concatenate,Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import Add, Dense, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU, PReLU, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'DPCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.h5'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')\
            if embedding != 'random' else None                                       # 预训练词向量

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_len = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.shape[1]\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.l2 = 0.0000032
        self.pooling_size_strides = [3, 2]
        self.dropout_spatial = 0.2
        self.activation_conv = 'linear'
        self.layer_repeats = 5
        self.full_connect_unit = 256

class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.embedding = Embedding(input_dim=self.config.embedding_pretrained.shape[0], output_dim=self.config.embedding_pretrained.shape[1],
                                   input_length=self.config.max_len, weights=[self.config.embedding_pretrained], trainable=True)

        self.embedding_output_spatial = SpatialDropout1D(config.dropout_spatial)

        # 首先是 region embedding 层
        self.conv_1 = Conv1D(self.config.num_filters,
                        kernel_size=1,
                        padding='SAME',
                        kernel_regularizer=l2(self.config.l2),
                        bias_regularizer=l2(self.config.l2),
                        activation=self.config.activation_conv,
                        )
        # 全连接层
        self.fc = Dense(self.config.full_connect_unit, activation='linear')
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(self.config.dropout)
        self.out_put = Dense(units=self.config.num_classes, activation='softmax')

    def block_and_layer(self, embedding_output_spatial,conv_1_prelu, layer_repeats):
        block = None
        layer_curr = 0
        for i in range(layer_repeats):
            if i == 0:  # 第一层输入用embedding输出的结果作为输入
                block = self.ResCNN(embedding_output_spatial)
                block_add = Add()([block, conv_1_prelu])
                block = MaxPooling1D(pool_size=self.config.pooling_size_strides[0],
                                     strides=self.config.pooling_size_strides[1])(block_add)
            elif layer_repeats - 1 == i:  # 最后一次repeat用GlobalMaxPooling1D
                block_last = self.ResCNN(block)
                # ResNet(shortcut连接|skip连接|residual连接), 这里是shortcut连接. 恒等映射, block+f(block)
                block_add = Add()([block_last, block])
                block = GlobalMaxPooling1D()(block_add)
                break
            else:  # 中间层 repeat
                block_mid = self.ResCNN(block)
                block_add = Add()([block_mid, block])
                block = MaxPooling1D(pool_size=self.config.pooling_size_strides[0],
                                     strides=self.config.pooling_size_strides[1])(block_add)
        return block

    def ResCNN(self, x):
        """
        repeat of two conv
        :param x: tensor, input shape
        :return: tensor, result of two conv of resnet
        """
        # pre-activation
        # x = PReLU()(x)
        x = Conv1D(self.config.num_filters,
                                kernel_size=1,
                                padding='SAME',
                                kernel_regularizer=l2(self.config.l2),
                                bias_regularizer=l2(self.config.l2),
                                activation=self.config.activation_conv,
                                )(x)
        x = BatchNormalization()(x)
        #x = PReLU()(x)
        x = Conv1D(self.config.num_filters,
                                kernel_size=1,
                                padding='SAME',
                                kernel_regularizer=l2(self.config.l2),
                                bias_regularizer=l2(self.config.l2),
                                activation=self.config.activation_conv,
                                )(x)
        x = BatchNormalization()(x)
        # x = Dropout(self.dropout)(x)
        x = PReLU()(x)
        return x

    def createModel(self, input):
        main_input = Input(shape=input, dtype='float64')
        x = self.embedding(main_input)
        embedding_output_spatial = self.embedding_output_spatial(x)
        x = self.conv_1(embedding_output_spatial)
        conv_1_prelu = PReLU()(x)
        x = self.block_and_layer(embedding_output_spatial, conv_1_prelu, self.config.layer_repeats)
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        out_put = self.out_put(x)
        model = tf.keras.Model(inputs=main_input, outputs=out_put)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
