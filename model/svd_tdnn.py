import tensorflow as tf
from model.pooling import statistics_pooling, self_attention, ghost_vlad
from model.common import prelu, shape_list
from collections import OrderedDict
from six.moves import range
from misc.utils import Params
import numpy as np
import re

# train with tdnn_asoftmax_m4_linear_bn_1e-2.json
json_path = 'F:/BOOKS/untitled5/tf-kaldi-speaker-master/egs/voxceleb/v1/nnet_conf/tdnn_asoftmax_m4_linear_bn_1e-2.json'
nnet_path = 'F:/BOOKS/untitled5/nnet/'
nname = 'tdnn_svd6'
nsplitname = 'tdnn/tdnn6_dense/'


def tdnn_svd6(features, params, is_training=None, reuse_variables=None, aux_features=None):
    """Build a TDNN network.
    The structure is similar to Kaldi, while it uses bn+relu rather than relu+bn.
    And there is no dilation used, so it has more parameters than Kaldi x-vector.

    Args:
        features: A tensor with shape [batch, length, dim].
        params: Configuration loaded from a JSON.
        is_training: True if the network is used for training.
        reuse_variables: True if the network has been built and enable variable reuse.
        aux_features: Auxiliary features (e.g. linguistic features or bottleneck features).
    :return:
        features: The output of the last layer.
        endpoints: An OrderedDict containing output of every components. The outputs are in the order that they add to
                   the network. Thus it is convenient to split the network by a output name
    """
    # ReLU is a normal choice while other activation function is possible.
    mid_channels = 32
    relu = tf.nn.relu
    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        if params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu

    endpoints = OrderedDict()
    with tf.variable_scope("tdnn_svd6", reuse=reuse_variables):
        # Convert to [b, 1, l, d]
        features = tf.expand_dims(features, 1)

        # Layer 1: [-2,-1,0,1,2] --> [b, 1, l-4, 512]
        # conv2d + batchnorm + relu
        features = tf.layers.conv2d(features,
                                32,
                                (1, 5),
                                activation=None,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                name='tdnn1_conv')
        endpoints["tdnn1_conv"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn1_bn")
        endpoints["tdnn1_bn"] = features
        features = relu(features, name='tdnn1_relu')
        endpoints["tdnn1_relu"] = features

        # Layer 2: [-2, -1, 0, 1, 2] --> [b ,1, l-4, 512]
        # conv2d + batchnorm + relu
        # This is slightly different with Kaldi which use dilation convolution
        features = tf.layers.conv2d(features,
                                    32,
                                    (1, 5),
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                    name='tdnn2_conv')
        endpoints["tdnn2_conv"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn2_bn")
        endpoints["tdnn2_bn"] = features
        features = relu(features, name='tdnn2_relu')
        endpoints["tdnn2_relu"] = features

        # Layer 3: [-3, -2, -1, 0, 1, 2, 3] --> [b, 1, l-6, 512]
        # conv2d + batchnorm + relu
        # Still, use a non-dilation one
        features = tf.layers.conv2d(features,
                                    32,
                                    (1, 7),
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                    name='tdnn3_conv')
        endpoints["tdnn3_conv"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn3_bn")
        endpoints["tdnn3_bn"] = features
        features = relu(features, name='tdnn3_relu')
        endpoints["tdnn3_relu"] = features

        # Convert to [b, l, 512]
        features = tf.squeeze(features, axis=1)
        # The output of the 3-rd layer can simply be rank 3.
        endpoints["tdnn3_relu"] = features

        # Layer 4: [b, l, 512] --> [b, l, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="tdnn4_dense")
        endpoints["tdnn4_dense"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn4_bn")
        endpoints["tdnn4_bn"] = features
        features = relu(features, name='tdnn4_relu')
        endpoints["tdnn4_relu"] = features

        # Layer 5: [b, l, x]
        if "num_nodes_pooling_layer" not in params.dict:
            # The default number of nodes before pooling
            params.dict["num_nodes_pooling_layer"] = 1500

        features = tf.layers.dense(features,
                                   params.num_nodes_pooling_layer,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="tdnn5_dense")
        endpoints["tdnn5_dense"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn5_bn")
        endpoints["tdnn5_bn"] = features
        features = relu(features, name='tdnn5_relu')
        endpoints["tdnn5_relu"] = features

        # Pooling layer
        # If you add new pooling layer, modify this code.
        # Statistics pooling
        # [b, l, 1500] --> [b, x]
        if params.pooling_type == "statistics_pooling":
            features = statistics_pooling(features, aux_features, endpoints, params, is_training)
        elif params.pooling_type == "self_attention":
            features = self_attention(features, aux_features, endpoints, params, is_training)
        elif params.pooling_type == "ghost_vlad":
            features = ghost_vlad(features, aux_features, endpoints, params, is_training)
        # elif params.pooling_type == "aux_attention":
        #     features = aux_attention(features, aux_features, endpoints, params, is_training)
        else:
            raise NotImplementedError("Not implement %s pooling" % params.pooling_type)
        endpoints['pooling'] = features

        # Utterance-level network
        # Layer 6.0: --[b, 512] ++[b, mid_channels]
        features = tf.layers.dense(features,
                                   mid_channels,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='tdnn6.0_dense')
        endpoints['tdnn6.0_dense'] = features

        # Layer 6.5: [mid_channels, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='tdnn6.5_dense')
        endpoints['tdnn6.5_dense'] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn6_bn")
        endpoints["tdnn6_bn"] = features
        features = relu(features, name='tdnn6_relu')
        endpoints["tdnn6_relu"] = features


        # Layer 7: [b, x]
        if "num_nodes_last_layer" not in params.dict:
            # The default number of nodes in the last layer
            params.dict["num_nodes_last_layer"] = 512

        features = tf.layers.dense(features,
                                   params.num_nodes_last_layer,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='tdnn7_dense')
        endpoints['tdnn7_dense'] = features

        if "last_layer_no_bn" not in params.dict:
            params.last_layer_no_bn = False

        if not params.last_layer_no_bn:
            features = tf.layers.batch_normalization(features,
                                                     momentum=params.batchnorm_momentum,
                                                     training=is_training,
                                                     name="tdnn7_bn")
            endpoints["tdnn7_bn"] = features

        if "last_layer_linear" not in params.dict:
            params.last_layer_linear = False

        if not params.last_layer_linear:
            # If the last layer is linear, no further activation is needed.
            features = relu(features, name='tdnn7_relu')
            endpoints["tdnn7_relu"] = features

    return features, endpoints

def abandon(u, s, v, dimension=1.0):
    '''
    This function is used to prune the dimensions of matrix after svd.
    Args:
        :param u, s, v: M = u @ s @ v.T
        :param dimension:
            if 0 <= dimension <= 1.0:
                now dimension means the ratio of maintained singular values
            elif dimension > 1:
                now dimension *exactly* means the dimension to maintain
        :return: pruned matrices u, s, v
    '''
    u = np.mat(u)
    s = np.mat(s)
    v = np.mat(v)
    tot = np.sum(s)
    part = 0
    k = 0
    if dimension > 1:
        k = int(dimension)
    else:
        for i in range(s.shape[1]):
            part += s[0, i]
            if part / tot >= dimension:
                k = i
                break
    u = u[:, :k]
    s = s[:, :k]
    v = v[:k, :]
    return u, s, v

def semi_orthogonal(mat):
    '''
    Remember, this function has nothing to tensorflow!
    Pay attention to your input! Don't give me anything like Tensor!
    :param mat: a matrix of np.array or np.mat maybe.
    :return: the semi_orthogonal mat
    '''
    def get_alpha(P):
        return np.sqrt(np.trace(P * P.T) / np.trace(P))
    def f(Q):
        return np.trace(Q * Q.T)
    I = np.identity(mat.shape[0])
    mat = np.mat(mat)
    nu = 1 / 8
    for i in range(20):
        P = mat * mat.T
        alpha = get_alpha(P)
        square_alpha = alpha ** 2
        Q = P - square_alpha * I
        print(f(Q), square_alpha)
        mat = mat - (4 * nu / square_alpha) * Q * mat
    return mat



if __name__ == '__main__':
    reader = tf.train.NewCheckpointReader(nnet_path + 'model-570000')
    u, s, v = np.linalg.svd(reader.get_tensor(nsplitname + 'kernel'))
    u, s, v = abandon(u, s, v, dimension=0.8)
    u = np.mat(u)
    s = np.mat(np.diag(np.array(s).squeeze()))
    v = np.mat(v)
    A = u * s
    B = v
    C = reader.get_tensor(nsplitname + 'bias')

    params = Params(json_path)
    x = tf.placeholder(tf.float32, [10, 175, 30], name='x')
    features, endpoints = tdnn_svd6(features=x, params=params, mid_channels=A.shape[1])
    init = tf.global_variables_initializer()
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for name in reader.get_variable_to_shape_map():
            if nsplitname in name or 'softmax' in name:
                continue
            herename = nname + re.match(r'(.+?)(\/.*)', name).group(2) + ':0'
            sess.run(tf.assign(graph.get_tensor_by_name(herename), reader.get_tensor(name)))

        sess.run(tf.assign(graph.get_tensor_by_name(nname + '/tdnn6.0_dense/kernel:0'), A))
        sess.run(tf.assign(graph.get_tensor_by_name(nname + '/tdnn6.5_dense/kernel:0'), B))
        sess.run(tf.assign(graph.get_tensor_by_name(nname + '/tdnn6.5_dense/bias:0'), C))

        saver.save(sess, 'svd/model')
