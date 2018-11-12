import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
from PIL import Image
import caffe

params_root = 'params/scope_model/'

def load_weights(net, weights=None):
    print(net.params.items())
    # load embedding
    embed_params = np.load(params_root + 'embed_matrix.npy')
    print(embed_params.shape)
    print(net.params['embed'][0].data.shape)
    net.params['embed'][0].data[...] = embed_params

    # load conv
    for i in range(2,5):
        # print(net.params['conv'+str(i)][0].data.shape)
        # print(net.params['conv'+str(i)][1].data.shape)
        conv_params = np.transpose(np.load(params_root + 'cnn_max_pool-' +
                    str(i) + '/W.npy'), [3, 2, 0, 1])
        bias_params = np.load(params_root + 'cnn_max_pool-' + str(i) +
                '/b.npy')
        # print(conv_params.shape)
        # print(bias_params.shape)
        net.params['conv'+str(i)][0].data[...] = conv_params 
        net.params['conv'+str(i)][1].data[...] = bias_params

    # load fc
    fc_params = np.load(params_root + 'dense/kernel.npy')
    bias_params = np.load(params_root + 'dense/bias.npy')
    print(fc_params.shape)
    print(net.params['dense'][0].data.shape)
    net.params['dense'][0].data[...] = np.transpose(fc_params, [1, 0])
    net.params['dense'][1].data[...] = bias_params

    # load scale
    beta_params = np.load(params_root + 'Batch_Normalization/beta.npy')
    gamma_params = np.load(params_root + 'Batch_Normalization/gamma.npy')

    # print net.params['scale'][0].data.shape
    # print beta_params.shape
    net.params['scale'][0].data[...] = gamma_params / 1.0005
    net.params['scale'][1].data[...] = beta_params

    # load bn
    moving_mean = np.load(params_root + 'Batch_Normalization/moving_mean.npy')
    moving_var = np.load(params_root + 'Batch_Normalization/moving_variance.npy')

    net.params['batchnorm'][0].data[...] = moving_mean
    net.params['batchnorm'][1].data[...] = moving_var
    net.params['batchnorm'][2].data[...] = 1.0 

    return net


if __name__ == '__main__':
    """parser = ArgumentParser()
    parser.add_argument("deploy_file", type=str, help="path to deployment file")
    parser.add_argument("npy_file", type=str, help="path to npy file")
    parser.add_argument("caffemodel_file", type=str, help="path to caffemodel file")
    args = parser.parse_args()

    deploy_file = args.deploy_file
    weights_file = args.npy_file
    caffemodel_file = args.caffemodel_file"""

    deploy_file = "./text_classification.prototxt"
    weights_file = "params/global_step.npy"
    caffemodel_file = "./text_classification.caffemodel"

    net = caffe.Net(deploy_file, caffe.TRAIN)
    weights = None
    net = load_weights(net, weights)
    net.save(caffemodel_file)

    print('Model Saved!')
