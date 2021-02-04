from tensorflow import keras
from tensorflow.keras import layers
from AttentionLayer import *


class AMCNN:
    def __init__(self, maxlen, embed_dim,words_count, filter_size, channel, mask_prob=0.7,att_reg=0.0001 ):
        """
        :param maxlen: Max length of sequence
        :param embed_dim: Embedding size of word embedding layer
        :param words_count:  Word count of Tokenizer
        :param filter_size:  Filter size of CNN layer
        :param channel: Number of Attention Layer Channels
        :param mask_prob: Masking proportion of Attention Layer(It only apply training model.)
        :param att_reg: L2 regularizer term of Attention Layer
        """
        self.maxlen = maxlen
        self.words_count = words_count
        self.embed_dim = embed_dim
        self.filter_size = filter_size
        self.channel = channel
        self.att_reg = att_reg
        num_filter = embed_dim // filter_size
        self.num_filters = list(range(1, num_filter + 1))
        self.mask_prob = mask_prob

    def build(self, emb_trainable=True, pre_emb=True, emb_weight=None):
        """
        :param emb_trainable: Define trainable of Embedding Layer
        :param pre_emb: Whether to use pre-trained embedding weights
        :param emb_weight: Pre-trained embedding weights
        :return:
        """
        inputs = layers.Input(shape=(self.maxlen,))
        pad_k = tf.expand_dims(tf.cast((inputs == 0), dtype=tf.float32) * -99999, axis=2)

        if pre_emb:
            emb_layer = layers.Embedding(self.words_count + 1, self.embed_dim, trainable=emb_trainable,
                                         weights=[emb_weight])
        else:
            emb_layer = layers.Embedding(self.words_count + 1, self.embed_dim, trainable=
            True)
        inputs_emb = emb_layer(inputs)

        # Bi-LSTM cell summary
        lstm_layer = layers.LSTM(self.embed_dim, return_sequences=True)
        bi_lstm = layers.Bidirectional(lstm_layer, merge_mode="ave")(inputs_emb)

        C_features, self.scalar_att, self.vector_att = AttentionLayer(self.embed_dim, self.embed_dim, self.channel, 0.0001,
                                                            self.mask_prob)(bi_lstm, pad_k)
        inputs_emb2 = tf.expand_dims(inputs_emb, axis=3)
        C_features = tf.concat([inputs_emb2, C_features], axis=3)

        # kim-cnn process
        pools = []
        for filter_sizes in self.num_filters:
            cnn_layers = layers.Conv2D(self.filter_size, kernel_size=(filter_sizes, self.embed_dim), activation="relu")
            cnn_out = cnn_layers(C_features)
            max_pools = layers.MaxPool2D(pool_size=(self.maxlen - filter_sizes + 1, 1))(cnn_out)
            max_pools = layers.Flatten()(max_pools)
            pools.append(max_pools)
        concated = layers.concatenate(pools)  # filter size x num_fiilters 수

        # Higy-way process
        gap_input_emb = layers.GlobalAvgPool1D()(inputs_emb)  # 임베딩 사이즈로 global average pooling
        trans_ = layers.Dense(self.embed_dim, activation="sigmoid", use_bias=True)(gap_input_emb)
        carry_ = 1 - trans_
        gap_ = layers.Multiply()([trans_, gap_input_emb])
        concated_ = layers.Multiply()([carry_, concated])
        concated_ = layers.Add()([concated_, gap_])
        outputs = layers.Dense(1, activation="sigmoid")(concated_)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

    def load_weights(self, path):
        self.model.load_weights(path)
        print("Load Weights Compelete!")
