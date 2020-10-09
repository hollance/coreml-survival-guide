# This is the source code from the chapter "Cleaning Up a Converted Model".

import tfcoreml
import coremltools as ct
from helpers import get_nn


def find_layer_index(name):
    for i, layer in enumerate(nn.layers):
        if layer.name == name:
            return i
    return None


# Convert the TensorFlow model to Core ML.

input_path = "deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb"
output_path = "DeepLab.mlmodel"

input_tensor = "ImageTensor:0"
input_name = "ImageTensor__0"
output_tensor = "ResizeBilinear_3:0"

model = tfcoreml.convert(tf_model_path=input_path,
                         mlmodel_path=output_path,
                         output_feature_names=[output_tensor],
                         input_name_shape_dict={input_tensor: [1, 513, 513, 3]},
                         image_input_names=input_name)

# Fill in the descriptions and metadata.

spec = model._spec

spec.description.metadata.versionString = "v1.0"
spec.description.metadata.shortDescription = "DeepLab v3+ on MobileNet v2"
spec.description.metadata.author = "https://github.com/tensorflow/models/tree/master/research/deeplab"
spec.description.metadata.license = "Apache License"

# Rename inputs and outputs.

old_input_name = "ImageTensor__0"
new_input_name = "image"
old_output_name = "ResizeBilinear_3__0"
new_output_name = "scores"

spec.description.input[0].name = new_input_name
spec.description.input[0].shortDescription = "Input image"
spec.description.output[0].name = new_output_name
spec.description.output[0].shortDescription = "Segmentation map"

nn = get_nn(spec)

for i in range(len(nn.layers)):
    if len(nn.layers[i].input) > 0: 
        if nn.layers[i].input[0] == old_input_name:
            nn.layers[i].input[0] = new_input_name
    if len(nn.layers[i].output) > 0: 
        if nn.layers[i].output[0] == old_output_name:
            nn.layers[i].output[0] = new_output_name

spec.neuralNetwork.preprocessing[0].featureName = new_input_name

# Remove the second-to-last layer.

nn.layers[-1].input[0] = nn.layers[-2].input[0]
del nn.layers[-2]

# Remove the unused layers at the beginning.

resize_layer = nn.layers[find_layer_index("ResizeBilinear:0")]
multiply_layer = nn.layers[find_layer_index("mul_1:0")]
multiply_layer.input[0] = resize_layer.input[0]

del nn.layers[find_layer_index("ResizeBilinear:0")]
del nn.layers[find_layer_index("negated_Reshape:0_sub_2:0")]
del nn.layers[find_layer_index("sub_2:0")]
del nn.layers[find_layer_index("Pad:0")]
del nn.layers[find_layer_index("Reshape/tensor:0")]
del nn.layers[find_layer_index("add_2:0")]

# Replace preprocessing layers.

nn.preprocessing[0].scaler.channelScale = 1/127.5
nn.preprocessing[0].scaler.redBias = -1.0
nn.preprocessing[0].scaler.greenBias = -1.0
nn.preprocessing[0].scaler.blueBias = -1.0

conv_layer = nn.layers[find_layer_index("MobilenetV2/Conv/Conv2D:0")]
conv_layer.input[0] = multiply_layer.input[0]

del nn.layers[find_layer_index("mul_1:0")]
del nn.layers[find_layer_index("sub_7:0")]

# Add an argmax layer at the end.

new_layer = nn.layers.add()
new_layer.name = "argmax"
params = ct.proto.NeuralNetwork_pb2.ReduceLayerParams
new_layer.reduce.mode = params.ARGMAX
new_layer.reduce.axis = params.C

new_layer.output.append(nn.layers[-2].output[0])
nn.layers[-2].output[0] = nn.layers[-2].name + "_output"
new_layer.input.append(nn.layers[-2].output[0])

# Fix up the output shape and make it INT32.

del spec.description.output[0].type.multiArrayType.shape[0]
spec.description.output[0].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32

# Convert weights to 16 bit floats and save the model.

spec = ct.utils.convert_neural_network_spec_weights_to_fp16(spec)
ct.models.utils.save_spec(spec, "DeepLabClean.mlmodel")
