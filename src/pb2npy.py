import os
import sys
import numpy as np
import tensorflow as tf

from PIL import Image
from argparse import ArgumentParser
from collections import OrderedDict


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    return graph


if __name__ == '__main__':
    #parser = ArgumentParser()
    #parser.add_argument("--frozen_model_filename", type=str, help="path to frozen model file")
    #parser.add_argument("--npy_filename", type=str, help="path to saved npy")
    #args = parser.parse_args()
    #frozen_model_filename = args.frozen_model_filename
    #npy = args.npy_filename
    frozen_model_filename = "./2Class_ResizeBilinear_3.pb"
    npy = "./try.npy"
    graph = load_graph(frozen_model_filename)

    inputs = graph.get_tensor_by_name('import/image:0')
    sigmoid = graph.get_tensor_by_name('import/ResizeBilinear_3:0')

    node = graph.get_tensor_by_name('import/xception_65/entry_flow/conv1_1/BatchNorm/moving_variance/read:0')
    print(node)


    old2new_map = OrderedDict()

    constant_values = OrderedDict()
    ops = graph.get_operations()
    with tf.Session(graph=graph) as sess:
        print(sess.run(node))
        exit()
        names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        for name in names:
            print(name)
        print('-------------------------------------------------------------------')
        constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
        for constant_op in constant_ops:
            constant_values[constant_op.name] = sess.run(constant_op.outputs[0])
            print(constant_op.name)
        print('-------------------------------------------------------------------')

    weights = OrderedDict()
    for key, value in constant_values.items():
        if len(value.shape) == 0:
            continue
        key = key.replace('import/', '')
        key = key.replace('xception_65', 'xception')
        key = key.replace('shortcut', 'match_conv')
        key = key.replace('_depthwise', '/dw_conv')
        key = key.replace('_pointwise', '/sep_conv')
        key = key.replace('BatchNorm', 'bn')
        key = key.lower()
        key = key.replace('conv2d_', 'conv')
        key = key.replace('logits/conv1c_1x1', 'fc')
        key = key.replace('depthwise_', '')
        weights[key] = value
        print(key)

    print("==================================================")
    key_value = [(key, value) for key, value in weights.items()]
    epsilon = 1e-3
    new_key_value = []
    for i in range(0, len(key_value)):
        layer_type = key_value[i][0].split('/')[-2]
        print(key_value[i][0])
        if layer_type == 'semantic':
            w_name = layer_type + "/weights"
            b_name = layer_type + "/bies"
            w = key_value[i][1]
            b = key_value[i+1][1]
            print(w_name); print(w.shape); print(b_name); print(b.shape)
            new_key_value.append((w_name, w))
            new_key_value.append((b_name, b))
            print('----------------------------------')
            break

        if (layer_type.find('bn') >= 0 or layer_type.find('scale') >= 0
                or layer_type.find('conv') >= 0
                or layer_type.find('aspp') >= 0 or layer_type == 'image_pooling' or layer_type.find('projection') >= 0):
            w = key_value[i][1]
            if layer_type.find('dw') >= 0:
                w = np.transpose(w, [0, 1, 3, 2])
            print(key_value[i][0], w.shape)
            new_key_value.append((key_value[i][0], w))
            print('----------------------------------')

        """if layer_type.find('conv') < 0 and layer_type != 'dw' and layer_type != 'sep'\
                and layer_type != 'concat_projection' and layer_type != 'feature_projection0':
            print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
            continue
        w = key_value[i][1]
        #if key_value[i][0].find('dw')>=0:
        #    w = np.transpose(w, [0, 1, 3, 2])
        if key_value[i+1][0].find('spacetobatchnd') >= 0:
            i += 4

        if key_value[i+1][0].find('beta')>=0:
            print(key_value[i+1][0], 'beta')
            beta = key_value[i+1][1]
            print(key_value[i+2][0], 'gamma')
            gamma = key_value[i+2][1]
        else:
            print(key_value[i+1][0], 'gamma')
            gamma = key_value[i+1][1]
            print(key_value[i+2][0], 'beta')
            beta = key_value[i+2][1]

        print(key_value[i+3][0], 'moving_mean')
        mean = key_value[i+3][1]

        print(key_value[i+4][0], 'moving_variance')
        variance = key_value[i+4][1]

        std = np.sqrt(variance+epsilon)
        #w = gamma / std * w
        #b = beta - gamma * mean / std
        w_name = key_value[i][0]
        #b_name = w_name.replace('weights', 'biases')

        print(w_name)
        print(w.shape)
        #print(b_name)
        #print(b.shape)

        new_key_value.append((w_name, w))
        #new_key_value.append((b_name, b))
        print('----------------------------------')"""

    for key, value in key_value:
        if key.find('fc')>=0:
            new_key_value.append((key, value))
            # print(key, value)

    update_weights = {key:value for key, value in new_key_value}

    print('=======================================================')
    final_weights = OrderedDict()
    for key, value in update_weights.items():
        if key.find('pad')>=0: continue
        if len(value.shape)==4:
            value = np.transpose(value, [3, 2, 0, 1])
        final_weights[key] = value
        print(key, value.shape)

    np.save(npy, final_weights)
