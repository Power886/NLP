# -*- coding: UTF-8 -*-
#微信公众号 小杜的nlp乐园 欢迎关注
#Author 小杜好好干
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Dropout, Flatten, MaxPool2D, concatenate,Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import Add, Dense, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU, PReLU, ReLU
from tensorflow.keras.layers import BatchNormalization,LayerNormalization


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Transformer'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
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
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.shape[1]\
            if self.embedding_pretrained is not None else 300           # 字向量维度

        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self, dim_model, num_head):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        self.dim_model = dim_model
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.wq = tf.keras.layers.Dense(dim_model)
        self.wk = tf.keras.layers.Dense(dim_model)
        self.wv = tf.keras.layers.Dense(dim_model)
        self.dense = Dense(dim_model)


    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_head, self.dim_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self,v,k,q,mask):
        batch_size = tf.shape(q)[0]
        Q = self.wq(q)
        K = self.wk(k)
        V = self.wv(v)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this

        scaled_attention,attention_weights = scaled_dot_product_attention(Q, K, V,mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dim_model))

        output = self.dense(concat_attention)
        return output,attention_weights

def point_wise_feed_forward_network(dim_model,hidden):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dim_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head)
        self.feed_forward = point_wise_feed_forward_network(dim_model, hidden)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x,training,mask):
        out,_ = self.attention(x,x,x,mask)
        attn_output = self.dropout1(out, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.feed_forward(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.config = config
        self.embedding = Embedding(input_dim=self.config.embedding_pretrained.shape[0],
                                   output_dim=self.config.embedding_pretrained.shape[1],
                                   input_length=self.config.pad_size, weights=[self.config.embedding_pretrained],
                                   trainable=True)
        self.postion_embedding = positional_encoding(config.n_vocab, config.dim_model)
        self.encoder_layers = [
            EncoderLayer(config.dim_model,config.num_head,config.hidden,config.dropout)
            for _ in range(config.num_encoder)]
        self.dropout = Dropout(self.config.dropout)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        mask = create_padding_mask(x)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.config.dim_model, tf.float32))
        x += self.postion_embedding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.config.num_encoder):
            x = self.encoder_layers[i](x, training, mask)
        return x

class Transformer(tf.keras.models.Model):
    def __init__(self,config):
        super(Transformer,self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.ffn_out = tf.keras.layers.Dense(config.num_classes, activation='softmax')
        self.dropout1 = tf.keras.layers.Dropout(config.dropout)

    def __call__(self,inp,training):
        enc_output = self.encoder(inp, training)
        out_shape = self.config.pad_size * self.config.embed
        enc_output = tf.reshape(enc_output, [-1, out_shape])
        ffn = self.dropout1(enc_output, training=training)
        ffn_out = self.ffn_out(ffn)
        model = tf.keras.Model(inputs=inp, outputs=ffn_out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model




# if __name__ == '__main__':
#     embedding = 'embedding_SougouNews.npz'
#     config = Config('../DuNews',embedding)
#     sample_encoder = Transformer(config)
#     # result = sample_encoder(tf.random.uniform((64, 30)),training=False, mask=None)
#     model = sample_encoder(Input(shape=(config.pad_size,), dtype='float64'), training=True)
#     model.summary()
    # sample_encoder_output = sample_encoder(tf.random.uniform((64, 30)),
    #                                        training=True)

    # print(config.embedding_pretrained.shape)  # (batch_size, input_seq_len, d_model)


