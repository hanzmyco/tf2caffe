import caffe
import numpy as np

caffe_model = 'text_classification.caffemodel'
deploy_proto = 'text_classification.prototxt'

net = caffe.Net(deploy_proto, caffe_model, caffe.TEST)

data = np.loadtxt('/data2/mycohzhang/source/text_classification/data/liulanqi_ads_all/test/data.0', dtype=np.int32)
result = np.loadtxt('/data2/mycohzhang/source/text_classification/data/liulanqi_ads_all/test/result.0', dtype=np.int32)
label = np.loadtxt('/data2/mycohzhang/source/text_classification/data/liulanqi_ads_all/test/label.0', dtype=np.int32)
print(len(data))
print(data[0])
print(data.shape)
count = 0
bcount = 0


for i in range(len(data)):
    net.blobs['data'].data[...] = data[i].reshape((1, 1, 20))
    out = net.forward() 
    if result[i] == int(out['argmax'][0][0][0]):
        count += 1
    if label[i] == int(out['argmax'][0][0][0]):
        bcount += 1
    print(i, result[i], int(out['argmax'][0][0][0]), label[i])

print (float(count) / len(result))
print (float(bcount) / len(result))
