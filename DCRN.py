'''
Descripttion: DCRN模型定义
Version: 1.0
Date: 2021-03-31 18:50:37
LastEditTime: 2021-05-05 15:25:45
'''
import numpy as np
import config
import tensorflow as tf
class ResidualBlock(tf.keras.layers.Layer):
    '''定义残差块 '''
    def __init__ (
            self,
            input_dim,
            residual_layers, 
            residual_layers_activation,
            dropout_residual,
        ):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.residual_layers = residual_layers
        self.output_dim = input_dim
        self.residual_layers_activation = residual_layers_activation
        self.dropout_residual = dropout_residual
        self._weights = self._initialize_weights()
        
    def _initialize_weights(self):
        weights = dict()
        # residual layers 残差网络的第一层隐藏层的权重
        glorot = np.sqrt(2.0/(self.input_dim + self.residual_layers[0]))

        weights['residual_layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(self.input_dim,self.residual_layers[0])),dtype=tf.float32
        )
        weights['residual_bias_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(1,self.residual_layers[0])),dtype=tf.float32
        )

        # 递推地初始化residual_layer后续隐藏层的权重
        for i in range(1,len(self.residual_layers)):
            glorot = np.sqrt(2.0 / (self.residual_layers[i - 1] + self.residual_layers[i]))
            weights["residual_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.residual_layers[i - 1], self.residual_layers[i])),
                dtype=tf.float32)  # layers[i-1] * layers[i]
            weights["residual_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.residual_layers[i])),
                dtype=tf.float32)  # 1 * layer[i]


        # 第个残差块的最后一层神经网络，相当于输出层，输出维度为num_features
        glorot = np.sqrt(2.0 / (self.residual_layers[-1] + self.output_dim))
        weights["residual_layer_%d" % len(self.residual_layers)] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.residual_layers[-1], self.output_dim)),
            dtype=tf.float32)
        weights["residual_bias_%d" % len(self.residual_layers)] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.output_dim)),
            dtype=tf.float32)
        
        return weights

    def call(self, x):
        y_residual = tf.nn.dropout(x, self.dropout_residual[0])
        for i in range(len(self.residual_layers)):
            y_residual = tf.add(tf.matmul(y_residual,self._weights["residual_layer_%d" % i]), self._weights["residual_bias_%d" % i ])
            y_residual = self.residual_layers_activation(y_residual)
            y_residual = tf.nn.dropout(y_residual, self.dropout_residual[i+1])
        
        y_residual = tf.add(tf.matmul(y_residual,self._weights["residual_layer_%d" % len(self.residual_layers)]), self._weights["residual_bias_%d" % len(self.residual_layers)])
        # y_residual = self.residual_layers_activation(y_residual)
        # y_residual = tf.nn.dropout(y_residual, self.dropout_residual[1])
        y_residual = tf.add_n([x, y_residual])
        y_residual = self.residual_layers_activation(y_residual)
        y_residual = tf.nn.dropout(y_residual, self.dropout_residual[len(self.residual_layers)+1])
        return y_residual


class DCRN(tf.keras.Model):
    def __init__(self,
                 feature_size = 256,
                 deep_layers=[32, 32], 
                 dropout_deep=[0.5, 0.5, 0.5],
                 residual_layers=[32],#残差网络的隐藏层维度，不包括最后一层
                 residual_blocks_num = 3, ##残差块个数
                 dropout_residual=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 residual_layers_activation=tf.nn.relu,
                 random_seed=2021,
                 cross_layer_num=3):

        super(DCRN, self).__init__()
        assert(len(dropout_deep)==len(deep_layers)+1)
        assert(len(dropout_residual)==len(residual_layers)+2)

       # 所有特征数量feature_size，即数值型的特征数量
        self.feature_size = feature_size
        self.deep_layers = deep_layers  # 存储各层的神经元数量
        self.residual_layers = residual_layers
        self.residual_blocks_num = 3
        self.dropout_residual = dropout_residual
        self.cross_layer_num = cross_layer_num
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.residual_layers_activation = residual_layers_activation
        self.random_seed = random_seed
         # 这里的weights是一个字典，包括Cross和Deep部分的所有weight和bias
        self._weights = self._initialize_weights()
        # 这里的residual_layers包括residual_layers部分的weight和bias
        self._residual_layers = tf.keras.Sequential()
        for i in range(self.residual_blocks_num):
            self._residual_layers.add(ResidualBlock(self.feature_size, residual_layers, residual_layers_activation, dropout_residual))
    
    #初始化deep部分和cross部分的模型权重
    def _initialize_weights(self):
        weights = dict()

        #deep layers
        num_layer = len(self.deep_layers)
        glorot = np.sqrt(2.0/(self.feature_size + self.deep_layers[0]))

        # 初始化deep_layer第一层
        weights['deep_layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(self.feature_size,self.deep_layers[0])),dtype=tf.float32
        )
        weights['deep_bias_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=tf.float32
        )
        weights['batch_norm_scale_0']= tf.Variable(np.random.normal(loc=0, scale=0.01, size=(1,)), dtype=tf.float32)
        weights['batch_norm_offset_0' ] = tf.Variable(np.random.normal(loc=0, scale=0.01, size=(1,)), dtype=tf.float32)

        # 递推地初始化deep_layer后续层的权重
        for i in range(1,num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["deep_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=tf.float32)  # layers[i-1] * layers[i]
            weights["deep_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=tf.float32)  # 1 * layer[i]
            # weights['batch_norm_scale_%d' % i ]= tf.Variable(np.random.normal(loc=0, scale=0.01, size=(1,)), dtype=tf.float32)
            # weights['batch_norm_offset_%d' % i ] = tf.Variable(np.random.normal(loc=0, scale=0.01, size=(1,)), dtype=tf.float32)

        # 初始化cross_layer部分的权重
        for i in range(self.cross_layer_num):
            weights["cross_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.feature_size,1)),
                dtype=tf.float32)
            weights["cross_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.feature_size,1)),
                dtype=tf.float32)  # 1 * layer[i]

        # final concat projection layer
        # 合并cross、reisudal和deep最后一层的layer
        input_size = 2*self.feature_size + self.deep_layers[-1]

        weights['batch_norm_scale']= tf.Variable(np.random.normal(loc=0, scale=0.01, size=(1,)), dtype=tf.float32)
        weights['batch_norm_offset'] = tf.Variable(np.random.normal(loc=0, scale=0.01, size=(1,)), dtype=tf.float32)

        glorot = np.sqrt(2.0/(input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size, 1)),dtype=tf.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01, dtype=tf.float32),dtype=tf.float32)

        return weights

    # 前向传播过程
    def call(self, x, training=True):
        assert(len(x.shape)==2) #限制输入为二维张量
        # deep part 深度网络部分
        y_deep = tf.nn.dropout(x, self.dropout_deep[0]) 
        for i in range(0,len(self.deep_layers)):
            y_deep = tf.add(tf.matmul(y_deep,self._weights["deep_layer_%d" %i]), self._weights["deep_bias_%d"%i])
            mean , variance  = tf.nn.moments(y_deep, axes = 0)
            # y_deep = tf.nn.batch_normalization(y_deep, mean, variance, offset=self._weights['batch_norm_offset_%d' % i ], scale=self._weights['batch_norm_scale_%d' % i], variance_epsilon=1e-10, name=None)
            y_deep = self.deep_layers_activation(y_deep)
            y_deep = tf.nn.dropout(y_deep, self.dropout_deep[i+1])

        # 残差网络部分
        y_residual = self._residual_layers(x)

        # cross_part，特征交叉部分
        x0 = tf.reshape(x, (-1, self.feature_size, 1))
        x_l = x0
        for l in range(self.cross_layer_num):
            x_l = tf.tensordot(tf.matmul(x0, x_l, transpose_b=True),
                                self._weights["cross_layer_%d" % l],1) + self._weights["cross_bias_%d" % l] + x_l
        cross_network_out = tf.reshape(x_l, (-1, self.feature_size))

        # concat_part，将deep网络、residual网络和cross网络和的输出合并
        concat_input = tf.concat([y_deep, y_residual, cross_network_out], axis=1)

        #全连接之前batch_norm
        mean , variance = tf.nn.moments(concat_input, axes = 0)
        concat_input = tf.nn.batch_normalization(concat_input, mean, variance, offset=self._weights['batch_norm_offset'], scale=self._weights['batch_norm_scale'], variance_epsilon=1e-10, name=None)

        # 全连接
        out = tf.add(tf.matmul(concat_input,self._weights['concat_projection']),self._weights['concat_bias'])

        # 经由sigmoid输出概率
        out = tf.nn.sigmoid(out)

        # sigmoid的输出在x很小/很大的情况下会取到0和1，需要修正。
        out = tf.clip_by_value(out, 1e-10, 0.9999)
        return out

    # 每个batch的训练代码
    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            #训练模式的前向传播
            y_hat = self(X, training=True) 

            loss = self.compiled_loss(tf.reshape(y, shape=(-1,)), tf.reshape(y_hat, shape=(-1, )))

            loss = tf.reduce_mean(loss) #返回的是这个batch的loss向量，需要对其求和
        # unconnected_gradients为无法求导时返回的值，有none和zero可选择，默认为none
        # 这里建议用zero，否则后面可能会产生运算错误
        grads = tape.gradient(loss, self.trainable_variables, unconnected_gradients="zero") 
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

    #每个batch的test代码
    def test_step(self, data):
        X, y = data
        # 测试模式的前向传播
        y_hat = self(X, training=False)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}