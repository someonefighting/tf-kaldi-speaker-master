import tensorflow as tf
from model.pooling import statistics_pooling, self_attention, ghost_vlad
from model.common import prelu, shape_list
from collections import OrderedDict
from six.moves import range
import numpy as np
import json
import re
import torch

def tdnn_svd(features, params, is_training=None, reuse_variables=None, aux_features=None):
    """Build a TDNN network.
    The structure is similar to Kaldi, while it uses bn+relu rather than relu+bn.
    And there is no dilation used, so it has more parameters than Kaldi x-vector.

    Args:
        features: A tensor with shape [batch, length, dim].
        params: Configuration loaded from a JSON.
        svd_params: Configuration to point out which layers to be svd loaded from a JSON.
                    And it should be updated by function "update_mid_channels" before passing to "svdtdnn".
        is_training: True if the network is used for training.
        reuse_variables: True if the network has been built and enable variable reuse.
        aux_features: Auxiliary features (e.g. linguistic features or bottleneck features).
    :return:
        features: The output of the last layer.
        endpoints: An OrderedDict containing output of every components. The outputs are in the order that they add to
                   the network. Thus it is convenient to split the network by a output name
    """
    name = 'tdnn_svd'
    svd_json_path = '/data2/liry/test/tf-kaldi-speaker/model/hello.json'
    svd_params = Params(svd_json_path)
    

    assert svd_params.updated

    # ReLU is a normal choice while other activation function is possible.
    relu = tf.nn.relu

    for layer in svd_params.split:
        if svd_params.split[layer] and svd_params.mid_channels[layer] == -1:
            raise AttributeError('Please update the mid_channels of %s before construct the graph' % layer)

    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        if params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu

    endpoints = OrderedDict()
    with tf.variable_scope(name, reuse=reuse_variables):
        # Convert to [b, 1, l, d]
        features = tf.expand_dims(features, 1)

        # Layer 1: [-2,-1,0,1,2] --> [b, 1, l-4, 512]
        # conv2d + batchnorm + relu
        if svd_params.split['tdnn1_conv']:
            features = tf.layers.conv2d(features,
                                        svd_params.mid_channels['tdnn1_conv'],
                                        (1, 5),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            params.weight_l2_regularizer),
                                        name='tdnn1.0_conv',
                                        bias_initializer=tf.zeros_initializer())
            endpoints["tdnn1.0_conv"] = features
            features = tf.layers.conv2d(features,
                                        32,
                                        (1, 1),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            params.weight_l2_regularizer),
                                        name='tdnn1.5_conv')
            endpoints["tdnn1.5_conv"] = features
        else:
            features = tf.layers.conv2d(features,
                                        32,
                                        (1, 5),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            params.weight_l2_regularizer),
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
        if svd_params.split['tdnn2_conv']:
            features = tf.layers.conv2d(features,
                                        svd_params.mid_channels['tdnn2_conv'],
                                        (1, 5),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            params.weight_l2_regularizer),
                                        name='tdnn2.0_conv',
                                        bias_initializer=tf.zeros_initializer())
            endpoints["tdnn2.0_conv"] = features
            features = tf.layers.conv2d(features,
                                        32,
                                        (1, 1),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            params.weight_l2_regularizer),
                                        name='tdnn2.5_conv')
            endpoints["tdnn2.5_conv"] = features
        else:
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
        if svd_params.split['tdnn3_conv']:
            features = tf.layers.conv2d(features,
                                        svd_params.mid_channels['tdnn3_conv'],
                                        (1, 7),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            params.weight_l2_regularizer),
                                        name='tdnn3.0_conv',
                                        bias_initializer=tf.zeros_initializer())
            endpoints["tdnn3.0_conv"] = features
            features = tf.layers.conv2d(features,
                                        32,
                                        (1, 1),
                                        activation=None,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            params.weight_l2_regularizer),
                                        name='tdnn3.5_conv')
            endpoints["tdnn3.5_conv"] = features
        else:
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
        if svd_params.split['tdnn4_dense']:
            features = tf.layers.dense(features,
                                       svd_params.mid_channels['tdnn4_dense'],
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           params.weight_l2_regularizer),
                                       name="tdnn4.0_dense",
                                       bias_initializer=tf.zeros_initializer())
            endpoints["tdnn4.0_dense"] = features
            features = tf.layers.dense(features,
                                       512,
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           params.weight_l2_regularizer),
                                       name="tdnn4.5_dense")
            endpoints["tdnn4.5_dense"] = features
        else:
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

        if svd_params.split['tdnn5_dense']:
            features = tf.layers.dense(features,
                                       svd_params.mid_channels['tdnn5_dense'],
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           params.weight_l2_regularizer),
                                       name="tdnn5.0_dense",
                                       bias_initializer=tf.zeros_initializer())
            endpoints["tdnn5.0_dense"] = features
            features = tf.layers.dense(features,
                                       params.num_nodes_pooling_layer,
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           params.weight_l2_regularizer),
                                       name="tdnn5.5_dense")
            endpoints["tdnn5.5_dense"] = features
        else:
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
        # Layer 6: [b, 512]
        if svd_params.split['tdnn6_dense']:
            features = tf.layers.dense(features,
                                       svd_params.mid_channels['tdnn6_dense'],
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           params.weight_l2_regularizer),
                                       name='tdnn6.0_dense',
                                       bias_initializer=tf.zeros_initializer())
            endpoints['tdnn6.0_dense'] = features
            features = tf.layers.dense(features,
                                       512,
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           params.weight_l2_regularizer),
                                       name='tdnn6.5_dense')
            endpoints['tdnn6.5_dense'] = features
        else:
            features = tf.layers.dense(features,
                                       512,
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                       name='tdnn6_dense')
            endpoints['tdnn6_dense'] = features
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

        if svd_params.split['tdnn7_dense']:
            features = tf.layers.dense(features,
                                       svd_params.mid_channels['tdnn7_dense'],
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           params.weight_l2_regularizer),
                                       name='tdnn7.0_dense',
                                       bias_initializer=tf.zeros_initializer())
            endpoints['tdnn7.0_dense'] = features
            features = tf.layers.dense(features,
                                       params.num_nodes_last_layer,
                                       activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           params.weight_l2_regularizer),
                                       name='tdnn7.5_dense')
            endpoints['tdnn7.5_dense'] = features
        else:
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


def update_mid_channels_in_json(reader, svd_params):
    '''
    This function is used to calculate the mid-channel in svd_params after appointing "dimension".
    Please ensure that all your conv-layers are named with 'conv',
    and dense-layers are named with 'dense'.
    Args:
        :param reader: tf.train.NewCheckpointReader()
        :param svd_params: Configuration to point out which layers to be svd loaded from a JSON.
    '''
    if reader == None:
        reader = tf.train.NewCheckpointReader('/data2/liry/test/tf-kaldi-speaker/egs/voxceleb/v1/exp_32/xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2/nnet/model-1410000')
    svd_params.updated = True
    ancestor_name = 'tdnn/'
    for layer in svd_params.split:
        if svd_params.split[layer]:
            M = reader.get_tensor(ancestor_name + layer + '/kernel')
            if 'dense' in layer:
                M = M
            elif 'conv' in layer:
                M = M.reshape(-1, M.shape[2])
            u, s, v = np.linalg.svd(M,
                full_matrices=False
            )
            u, s, v = abandon(u, s, v, svd_params.dimension[layer])
            svd_params.mid_channels[layer] = u.shape[1]


def svd_A_B(M, svd_params):
    u, s, v = np.linalg.svd(
        M,
        full_matrices=False
    )
    u, s, v = abandon(u, s, v, svd_params.mid_channels[layer_name])
    u = np.mat(u)
    s = np.mat(np.diag(np.array(s).squeeze()))
    v = np.mat(v)
    A = u * s
    B = v
    return np.array(A), np.array(B)


class Params():

    def __init__(self, json_path=None):
        if json_path != None:
            self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

# svd_params = Params('hello.json')
# update_mid_channels_in_json(None, svd_params)
# svd_params.save('hello.json')
if __name__ == '__main__':

   
    ancestor_path = 'F:/BOOKS/untitled5/nnet/model-570000'
    svd_path = 'ck33/model'
    svd_json_path = 'hello.json'
    reader = tf.train.NewCheckpointReader(ancestor_path)
    saver = tf.train.import_meta_graph(svd_path + '.meta')
    graph = tf.get_default_graph()

    init = tf.global_variables_initializer()
    svd_params = Params(svd_json_path)
    assert svd_params.updated
    network_name = svd_params.network_name
    update_mid_channels_in_json(reader, svd_params)

    with tf.Session() as sess:
        sess.run(init)
        for name in reader.get_variable_to_shape_map():
            layer_name = re.match(r'(.+?)\/(.*?)\/(.*)', name).group(2)
            if layer_name in svd_params.split and svd_params.split[layer_name]:
                continue
            if 'kernel' in name or 'softmax' in name:
                continue
            herename = network_name + re.match(r'(.+?)(\/.*)', name).group(2) + ':0'
            sess.run(tf.assign(graph.get_tensor_by_name(herename), reader.get_tensor(name)))
        sess.run(tf.assign(graph.get_tensor_by_name('softmax/output/kernel:0'), reader.get_tensor('softmax/output/kernel')))
        for layer_name in svd_params.split:
            if svd_params.split[layer_name]:
                prename = 'tdnn/' + layer_name + '/kernel'
                M = reader.get_tensor(prename)
                shape = M.shape
                if 'dense' in layer_name:
                    M = M
                elif 'conv' in layer_name:
                    M = M.reshape(-1, M.shape[3])
                A, B = svd_A_B(M, svd_params)
                herename1 = network_name + '/' + layer_name[:5] + '.0' + layer_name[5:] + '/kernel:0'
                herename2 = network_name + '/' + layer_name[:5] + '.5' + layer_name[5:] + '/kernel:0'

                if 'dense' in layer_name:
                    sess.run(tf.assign(graph.get_tensor_by_name(herename1), A))
                    sess.run(tf.assign(graph.get_tensor_by_name(herename2), B))
                elif 'conv' in layer_name:
                    Aa = graph.get_tensor_by_name(herename1)
                    Bb = graph.get_tensor_by_name(herename2)
                    sess.run(tf.assign(Aa, A.reshape(Aa.shape)))
                    sess.run(tf.assign(Bb, B.reshape(Bb.shape)))

                prename = 'tdnn/' + layer_name + '/bias'
                herename = network_name + '/' + layer_name[:5] + '.5' + layer_name[5:] + '/bias:0'
                sess.run(tf.assign(graph.get_tensor_by_name(herename), reader.get_tensor(prename)))

        saver.save(sess, svd_path)
