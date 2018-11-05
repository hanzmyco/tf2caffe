import os
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from collections import OrderedDict

caffe_root = '/root/caffe1s'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe1s

def load_weight_map(path):
    weights_map = OrderedDict()
    file = open(path)
    for line in file.readlines():
        weights_map[line.split(',')[0]] = line.split(',')[1].replace('\n', '')
    return weights_map


def load_weights(net, weights):
    print(net.params)
    print(weights.keys())
    weights_map = load_weight_map('./weight_map_file.txt')
    for key, value in net.params.items():
        print(key)
        if key.find('bn') >= 0:
            key_map = key.replace('_bn', '')
            key_mean = weights_map[key_map] + '/bn/moving_mean'
            print(key_mean, value[0].data.shape)
            net.params[key][0].data[...] = weights[key_mean]
            print(key_mean, weights[key_mean].shape)

            key_variance = weights_map[key_map] + '/bn/moving_variance'
            print(key_variance, value[1].data.shape)
            net.params[key][1].data[...] = weights[key_variance]
            print(key_variance, weights[key_variance].shape)

            net.params[key][2].data[...] = 1.0

        elif key.find('scale') >= 0:
            key_map = key.replace('_scale', '')
            key_gamma = weights_map[key_map] + '/bn/gamma'
            print(key_gamma, value[0].data.shape)
            net.params[key][0].data[...] = weights[key_gamma]
            print(key_gamma, weights[key_gamma].shape)

            key_beta = weights_map[key_map] + '/bn/beta'
            print(key_beta, value[1].data.shape)
            net.params[key][1].data[...] = weights[key_beta]
            print(key_beta, weights[key_beta].shape)

        elif key.find('conv') >= 0 or key.find('fc') >= 0:
            key_weights = weights_map[key] + '/weights'
            key_biases = weights_map[key] + '/biases'
            print(key_weights, value[0].data.shape)
            net.params[key][0].data[...] = weights[key_weights]
            print(key_weights, value[0].data.shape)

            if key_biases in weights:
                print(key_biases, value[1].data.shape)
                net.params[key][1].data[...] = weights[key_biases]
                print(key_biases, value[1].data.shape)
        else:
            print('Unknown layer')
            continue
            raise Exception('Unknown layer')
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

    deploy_file = "./Deeplab_V3_Xception_cyBN_final.prototxt"
    weights_file = "./try.npy"
    caffemodel_file = "./Deeplab_V3_Xception_cy_final.caffemodel"

    net = caffe1s.Net(deploy_file, caffe1s.TRAIN)
    #print(net.params)
    #exit()
    weights = np.load(weights_file).tolist()
    #try:
    net = load_weights(net, weights)
    #except:
    net.save(caffemodel_file)
    net.save(caffemodel_file)

    print('Model Saved!')
