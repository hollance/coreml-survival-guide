import numpy as np
import h5py

caffemodel_path = "cifar10_quick_iter_5000.caffemodel.h5"
f = h5py.File(caffemodel_path, "r")


def get_weights(layer_name):
    weights = f[layer_name + "/0"][...]
    
    # Transpose the weights for a convolutional layer.
    if weights.ndim == 4:
        weights = weights.transpose(2, 3, 1, 0)

    biases = f[layer_name + "/1"][...]
    return weights, biases


import coremltools
from coremltools.models import datatypes
from coremltools.models import neural_network

input_features = [ ("image", datatypes.Array(3, 32, 32)) ]
output_features = [ ("labelProbs", None) ]

builder = neural_network.NeuralNetworkBuilder(input_features, 
                                              output_features, 
                                              mode="classifier")

builder.set_pre_processing_parameters(image_input_names=["image"], 
                                      is_bgr=False,
                                      red_bias=-125.3,
                                      green_bias=-122.95,
                                      blue_bias=-113.87)

cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", 
                  "dog", "frog", "horse", "ship", "truck"]

builder.set_class_labels(class_labels=cifar10_labels,
                         predicted_feature_name="label",
                         prediction_blob="labelProbs")

W, b = get_weights("data/conv1")

builder.add_convolution(name="conv1",
                        kernel_channels=3,
                        output_channels=32,
                        height=5,
                        width=5,
                        stride_height=1,
                        stride_width=1,
                        border_mode="valid",
                        groups=1,
                        W=W,
                        b=b,
                        has_bias=True,
                        input_name="image",
                        output_name="conv1_output", 
                        padding_top=2,
                        padding_bottom=2,
                        padding_left=2,
                        padding_right=2)

builder.add_pooling(name="pool1",
                    height=3,
                    width=3,
                    stride_height=2,
                    stride_width=2,
                    layer_type="MAX",
                    padding_type="INCLUDE_LAST_PIXEL",
                    input_name="conv1_output",
                    output_name="pool1_output")

builder.add_activation(name="relu1",
                       non_linearity="RELU",
                       input_name="pool1_output",
                       output_name="relu1_output")

W, b = get_weights("data/conv2")

builder.add_convolution(name="conv2",
                        kernel_channels=32,
                        output_channels=32,
                        height=5,
                        width=5,
                        stride_height=1,
                        stride_width=1,
                        border_mode="valid",
                        groups=1,
                        W=W,
                        b=b,
                        has_bias=True,
                        input_name="relu1_output",
                        output_name="conv2_output", 
                        padding_top=2,
                        padding_bottom=2,
                        padding_left=2,
                        padding_right=2)

# NOTE: ReLU comes before the pooling here

builder.add_activation(name="relu2",
                       non_linearity="RELU",
                       input_name="conv2_output",
                       output_name="relu2_output")

builder.add_pooling(name="pool2",
                    height=3,
                    width=3,
                    stride_height=2,
                    stride_width=2,
                    layer_type="AVERAGE",
                    padding_type="INCLUDE_LAST_PIXEL",
                    input_name="relu2_output",
                    output_name="pool2_output")

W, b = get_weights("data/conv3")

builder.add_convolution(name="conv3",
                        kernel_channels=32,
                        output_channels=64,
                        height=5,
                        width=5,
                        stride_height=1,
                        stride_width=1,
                        border_mode="valid",
                        groups=1,
                        W=W,
                        b=b,
                        has_bias=True,
                        input_name="pool2_output",
                        output_name="conv3_output", 
                        padding_top=2,
                        padding_bottom=2,
                        padding_left=2,
                        padding_right=2)

builder.add_activation(name="relu3",
                       non_linearity="RELU",
                       input_name="conv3_output",
                       output_name="relu3_output")

builder.add_pooling(name="pool3",
                    height=3,
                    width=3,
                    stride_height=2,
                    stride_width=2,
                    layer_type="AVERAGE",
                    padding_type="INCLUDE_LAST_PIXEL",
                    input_name="relu3_output",
                    output_name="pool3_output")

builder.add_flatten(name="flatten1", 
                    mode=0, 
                    input_name="pool3_output", 
                    output_name="flatten1_output")

W, b = get_weights("data/ip1")

builder.add_inner_product(name="ip1",
                          W=W,
                          b=b,
                          input_channels=1024,
                          output_channels=64,
                          has_bias=True,
                          input_name="flatten1_output",
                          output_name="ip1_output")

W, b = get_weights("data/ip2")

builder.add_inner_product(name="ip2",
                          W=W,
                          b=b,
                          input_channels=64,
                          output_channels=10,
                          has_bias=True,
                          input_name="ip1_output",
                          output_name="ip2_output")

builder.add_softmax(name="softmax",
                    input_name="ip2_output",
                    output_name="labelProbs")


import caffe_pb2
mean_image = caffe_pb2.BlobProto()
mean_image.ParseFromString(open("mean.binaryproto", "rb").read())
mean_image = np.array(mean_image.data)
builder.spec.neuralNetworkClassifier.preprocessing[0].meanImage.meanImage.extend(mean_image)


mlmodel = coremltools.models.MLModel(builder.spec)

mlmodel.short_description = "cifar10_quick"
mlmodel.author = "https://github.com/BVLC/caffe/tree/master/examples/cifar10"
mlmodel.license = "https://github.com/BVLC/caffe/blob/master/LICENSE"

mlmodel.input_description["image"] = "The input image"
mlmodel.output_description["labelProbs"] = "The predicted probabilities"
mlmodel.output_description["label"] = "The class with the highest score"

mlmodel.save("CIFAR10.mlmodel")


import PIL
img = PIL.Image.open("boat.jpg")
img = img.resize((32, 32), PIL.Image.BILINEAR)

prediction = mlmodel.predict({"image": img}, usesCPUOnly=True)
print(prediction)
