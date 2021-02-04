import tensorflow as tf

# Define Custom AttentionLayer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, input_units, output_unit, channel, reg, mask_prob, **kwargs, ):
        super(AttentionLayer, self).__init__(**kwargs)
        self.input_units = input_units
        self.output_unit = output_unit
        self.channel = channel
        self.reg = reg
        self.mask_prob = mask_prob

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_units': self.input_units,
            'output_unit': self.output_unit,
            "channel": self.channel,
            "reg": self.reg,
            "mask_prob": self.mask_prob
        })
        return config

    def build(self, input_shape):
        self.w_l_lst = []
        self.b_l_lst = []
        self.w_v_l1_lst = []
        self.w_v_l2_lst = []
        self.b_v_l_lst = []

        for i in range(self.channel):
            self.w_l_lst.append(self.add_weight("scalar_w_%s" % (i),
                                                initializer=tf.keras.initializers.glorot_normal(),
                                                regularizer=tf.keras.regularizers.l2(self.reg),
                                                shape=[self.input_units, self.output_unit]))

            self.b_l_lst.append(self.add_weight("scalar_b_%s" % (i),
                                                initializer=tf.zeros_initializer(),
                                                shape=[1, 1]))
            self.w_v_l1_lst.append(self.add_weight("vector_w1_%s" % (i),
                                                   initializer=tf.keras.initializers.glorot_normal(),
                                                   regularizer=tf.keras.regularizers.l2(self.reg),
                                                   shape=[int(self.output_unit), 1]))

            self.w_v_l2_lst.append(self.add_weight("vector_w2_%s" % (i),
                                                   initializer=tf.keras.initializers.glorot_normal(),
                                                   regularizer=tf.keras.regularizers.l2(self.reg),
                                                   shape=[self.input_units, int(self.output_unit)]))

            self.b_v_l_lst.append(self.add_weight("vector_b_%s" % (i),
                                                  initializer=tf.zeros_initializer(),
                                                  shape=[1, 1, int(self.output_unit)]))

    def call(self, inputs, pad_k):
        Mask_layer = tf.keras.layers.Dropout(self.mask_prob)
        C_l_total = []
        scalar_att_lst = []
        vector_att_lst = []
        # Zero padding
        pad_k2 = (pad_k + 99999) / 99999
        maxlen = inputs.shape[1]
        for i in range(self.channel):
            # make association matrix
            # Scalar-attention
            mat_l = tf.matmul(inputs, self.w_l_lst[i])
            mat_lj = tf.tile(tf.expand_dims(mat_l, axis=1), [1, maxlen, 1, 1])
            mat_li = tf.tile(tf.expand_dims(inputs, axis=1), [1, maxlen, 1, 1])
            mat_li = tf.transpose(mat_li, [0, 2, 1, 3])
            # M_l = tf.nn.tanh(tf.reduce_sum((mat_li*mat_lj)/np.sqrt(self.output_unit),axis=3)+self.b_l_lst[i])
            M_l = tf.reduce_sum((mat_li * mat_lj), axis=3) + self.b_l_lst[i]  # /np.sqrt(self.output_unit)

            A_l = tf.nn.tanh(Mask_layer(M_l))

            s_lk = tf.reduce_sum(A_l, axis=2)
            s_lk = tf.expand_dims(s_lk, axis=2)
            score_lk = pad_k + s_lk  #
            a_l = tf.nn.softmax(score_lk, axis=1)
            scalar_att_lst.append(a_l)

            # vectorical attention
            score_l_bar = tf.nn.tanh(tf.matmul(inputs, self.w_v_l2_lst[i]) + self.b_v_l_lst[i])
            score_l_bar = tf.matmul(score_l_bar, self.w_v_l1_lst[i]) + pad_k
            a_l_bar = tf.nn.softmax(score_l_bar, axis=1)
            new_inputs = tf.tile(tf.expand_dims(tf.reduce_sum(inputs * a_l_bar, axis=1), axis=1), [1, maxlen, 1])
            vector_att_lst.append(a_l_bar)
            C_l = a_l * inputs + new_inputs * pad_k2
            # C_l = a_l* (a_l_bar*inputs)
            # C_l = a_l*inputs + a_l_bar*inputs

            C_l = tf.expand_dims(C_l, axis=3)
            C_l_total.append(C_l)
        C_features = tf.concat(C_l_total, axis=3)
        inputs2 = tf.expand_dims(inputs, axis=3)
        C_features = tf.concat([C_features, inputs2], axis=3)
        return C_features, scalar_att_lst, vector_att_lst  # feature,scal attention, vector attention