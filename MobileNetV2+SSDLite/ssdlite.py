# Converts the MobileNetV2+SSDLite model to Core ML.
#
# This script creates a pipeline with three models:
#   1. MobileNetV2 + SSDLite
#   2. A neural network that decodes the coordinate predictions using the anchor boxes.
#   3. Non-maximum suppression
#
# This is the model from the paper 'SSD: Single Shot MultiBox Detector' by Liu et al (2015),
# https://arxiv.org/abs/1512.02325, with MobileNetV2 as the backbone and depthwise separable
# convolutions for the SSD layers (also known as SSDLite).
#
# The version of the model used is ssdlite_mobilenet_v2_coco, downloaded from:
# http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
#
# It was originally trained with the TensorFlow Object Detection API:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
#
# The model expects input images of 300x300 pixels and detects objects from the COCO dataset.
# The COCO class labels are included in the mlmodel file's metadata.
#
# NOTE: The conversion script reads from saved_model.pb, not from frozen_inference_graph.pb.
# (Using the frozen graph gives an error, "ValueError: Graph has cycles".)
#
# Tested with Python 3.6.5, Tensorflow 1.7.0, coremltools 2.0, tfcoreml 0.3.0.
# WARNING: If you run this with a different version of tfcoreml, some things may not work!
#
# See also: https://github.com/tf-coreml/tf-coreml/blob/master/examples/ssd_example.ipynb

import numpy as np

import tensorflow as tf
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

import tfcoreml
import coremltools


# From where to load the saved_model.pb file.
saved_model_path = "ssdlite_mobilenet_v2_coco_2018_05_09/saved_model"

# Where to save the final Core ML model file.
coreml_model_path = "ObjectDetection/ObjectDetection/MobileNetV2_SSDLite.mlmodel"

# The number of predicted classes, excluding background.
num_classes = 90

# The number of predicted bounding boxes.
num_anchors = 1917

# Size of the expected input image.
input_width = 300
input_height = 300


# =============================
# PART 1: MobileNetV2 + SSDLite
# =============================

# Temporary file. You can delete this after the conversion is done.
frozen_model_file = "frozen_model.pb"

# Names of the interesting tensors in the graph. We use "Postprocessor/convert_scores"
# instead of "concat_1" because this already applies the sigmoid to the class scores.
input_node = "Preprocessor/sub"
bbox_output_node = "concat"
class_output_node = "Postprocessor/convert_scores"

input_tensor = input_node + ":0"
bbox_output_tensor = bbox_output_node + ":0"
class_output_tensor = class_output_node + ":0"


def load_saved_model(path):
    """Loads a saved model into a graph."""
    print("Loading saved_model.pb from '%s'" % path)
    the_graph = tf.Graph()
    with tf.Session(graph=the_graph) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], path)
    return the_graph


def optimize_graph(graph):
    """Strips unused subgraphs and save it as another frozen TF model."""
    gdef = strip_unused_lib.strip_unused(
            input_graph_def = graph.as_graph_def(),
            input_node_names = [input_node],
            output_node_names = [bbox_output_node, class_output_node],
            placeholder_type_enum = dtypes.float32.as_datatype_enum)

    with gfile.GFile(frozen_model_file, "wb") as f:
        f.write(gdef.SerializeToString())


# Load the original graph and remove anything we don't need.
the_graph = load_saved_model(saved_model_path)
optimize_graph(the_graph)

# Convert to Core ML model.
ssd_model = tfcoreml.convert(
    tf_model_path=frozen_model_file,
    mlmodel_path=coreml_model_path,
    input_name_shape_dict={ input_tensor: [1, input_height, input_width, 3] },
    image_input_names=input_tensor,
    output_feature_names=[bbox_output_tensor, class_output_tensor],
    is_bgr=False,
    red_bias=-1.0,
    green_bias=-1.0,
    blue_bias=-1.0,
    image_scale=2./255)

spec = ssd_model.get_spec()

# Rename the inputs and outputs to something more readable.
spec.description.input[0].name = "image"
spec.description.input[0].shortDescription = "Input image"
spec.description.output[0].name = "scores"
spec.description.output[0].shortDescription = "Predicted class scores for each bounding box"
spec.description.output[1].name = "boxes"
spec.description.output[1].shortDescription = "Predicted coordinates for each bounding box"

input_mlmodel = input_tensor.replace(":", "__").replace("/", "__")
class_output_mlmodel = class_output_tensor.replace(":", "__").replace("/", "__")
bbox_output_mlmodel = bbox_output_tensor.replace(":", "__").replace("/", "__")

for i in range(len(spec.neuralNetwork.layers)):
    if spec.neuralNetwork.layers[i].input[0] == input_mlmodel:
        spec.neuralNetwork.layers[i].input[0] = "image"
    if spec.neuralNetwork.layers[i].output[0] == class_output_mlmodel:
        spec.neuralNetwork.layers[i].output[0] = "scores"
    if spec.neuralNetwork.layers[i].output[0] == bbox_output_mlmodel:
        spec.neuralNetwork.layers[i].output[0] = "boxes"

spec.neuralNetwork.preprocessing[0].featureName = "image"

# For some reason the output shape of the "scores" output is not filled in.
spec.description.output[0].type.multiArrayType.shape.append(num_classes + 1)
spec.description.output[0].type.multiArrayType.shape.append(num_anchors)

# And the "boxes" output shape is (4, 1917, 1) so get rid of that last one.
del spec.description.output[1].type.multiArrayType.shape[-1]

# Convert weights to 16-bit floats to make the model smaller.
spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)

# Create a new MLModel from the modified spec and save it.
ssd_model = coremltools.models.MLModel(spec)
ssd_model.save(coreml_model_path)


# ================================
# PART 2: Decoding the coordinates
# ================================

def get_anchors(graph, tensor_name):
    """
    Computes the list of anchor boxes by sending a fake image through the graph.
    Outputs an array of size (4, num_anchors) where each element is an anchor box
    given as [ycenter, xcenter, height, width] in normalized coordinates.
    """
    image_tensor = graph.get_tensor_by_name("image_tensor:0")
    box_corners_tensor = graph.get_tensor_by_name(tensor_name)
    box_corners = sess.run(box_corners_tensor, feed_dict={image_tensor: np.zeros((1, input_height, input_width, 3))})

    # The TensorFlow graph gives each anchor box as [ymin, xmin, ymax, xmax]. 
    # Convert these min/max values to a center coordinate, width and height.
    ymin, xmin, ymax, xmax = np.transpose(box_corners)
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return np.stack([ycenter, xcenter, height, width])


# Read the anchors into a (4, 1917) tensor.
anchors_tensor = "Concatenate/concat:0"
with the_graph.as_default():
    with tf.Session(graph=the_graph) as sess:
        anchors = get_anchors(the_graph, anchors_tensor)
        assert(anchors.shape[1] == num_anchors)


from coremltools.models import datatypes
from coremltools.models import neural_network

# MLMultiArray inputs of neural networks must have 1 or 3 dimensions. 
# We only have 2, so add an unused dimension of size one at the back.
input_features = [ ("scores", datatypes.Array(num_classes + 1, num_anchors, 1)),
                   ("boxes", datatypes.Array(4, num_anchors, 1)) ]

# The outputs of the decoder model should match the inputs of the next
# model in the pipeline, NonMaximumSuppression. This expects the number
# of bounding boxes in the first dimension.
output_features = [ ("raw_confidence", datatypes.Array(num_anchors, num_classes)),
                    ("raw_coordinates", datatypes.Array(num_anchors, 4)) ]

builder = neural_network.NeuralNetworkBuilder(input_features, output_features)

# (num_classes+1, num_anchors, 1) --> (1, num_anchors, num_classes+1)
builder.add_permute(name="permute_scores",
                    dim=(0, 3, 2, 1),
                    input_name="scores",
                    output_name="permute_scores_output")

# Strip off the "unknown" class (at index 0).
builder.add_slice(name="slice_scores",
                  input_name="permute_scores_output",
                  output_name="raw_confidence",
                  axis="width",
                  start_index=1,
                  end_index=num_classes + 1)

# Grab the y, x coordinates (channels 0-1).
builder.add_slice(name="slice_yx",
                  input_name="boxes",
                  output_name="slice_yx_output",
                  axis="channel",
                  start_index=0,
                  end_index=2)

# boxes_yx / 10
builder.add_elementwise(name="scale_yx",
                        input_names="slice_yx_output",
                        output_name="scale_yx_output",
                        mode="MULTIPLY",
                        alpha=0.1)

# Split the anchors into two (2, 1917, 1) arrays.
anchors_yx = np.expand_dims(anchors[:2, :], axis=-1)
anchors_hw = np.expand_dims(anchors[2:, :], axis=-1)

builder.add_load_constant(name="anchors_yx",
                          output_name="anchors_yx",
                          constant_value=anchors_yx,
                          shape=[2, num_anchors, 1])

builder.add_load_constant(name="anchors_hw",
                          output_name="anchors_hw",
                          constant_value=anchors_hw,
                          shape=[2, num_anchors, 1])

# (boxes_yx / 10) * anchors_hw
builder.add_elementwise(name="yw_times_hw",
                        input_names=["scale_yx_output", "anchors_hw"],
                        output_name="yw_times_hw_output",
                        mode="MULTIPLY")

# (boxes_yx / 10) * anchors_hw + anchors_yx
builder.add_elementwise(name="decoded_yx",
                        input_names=["yw_times_hw_output", "anchors_yx"],
                        output_name="decoded_yx_output",
                        mode="ADD")

# Grab the height and width (channels 2-3).
builder.add_slice(name="slice_hw",
                  input_name="boxes",
                  output_name="slice_hw_output",
                  axis="channel",
                  start_index=2,
                  end_index=4)

# (boxes_hw / 5)
builder.add_elementwise(name="scale_hw",
                        input_names="slice_hw_output",
                        output_name="scale_hw_output",
                        mode="MULTIPLY",
                        alpha=0.2)

# exp(boxes_hw / 5)
builder.add_unary(name="exp_hw",
                  input_name="scale_hw_output",
                  output_name="exp_hw_output",
                  mode="exp")

# exp(boxes_hw / 5) * anchors_hw
builder.add_elementwise(name="decoded_hw",
                        input_names=["exp_hw_output", "anchors_hw"],
                        output_name="decoded_hw_output",
                        mode="MULTIPLY")

# The coordinates are now (y, x) and (height, width) but NonMaximumSuppression
# wants them as (x, y, width, height). So create four slices and then concat
# them into the right order.
builder.add_slice(name="slice_y",
                  input_name="decoded_yx_output",
                  output_name="slice_y_output",
                  axis="channel",
                  start_index=0,
                  end_index=1)

builder.add_slice(name="slice_x",
                  input_name="decoded_yx_output",
                  output_name="slice_x_output",
                  axis="channel",
                  start_index=1,
                  end_index=2)

builder.add_slice(name="slice_h",
                  input_name="decoded_hw_output",
                  output_name="slice_h_output",
                  axis="channel",
                  start_index=0,
                  end_index=1)

builder.add_slice(name="slice_w",
                  input_name="decoded_hw_output",
                  output_name="slice_w_output",
                  axis="channel",
                  start_index=1,
                  end_index=2)

builder.add_elementwise(name="concat",
                        input_names=["slice_x_output", "slice_y_output", 
                                     "slice_w_output", "slice_h_output"],
                        output_name="concat_output",
                        mode="CONCAT")

# (4, num_anchors, 1) --> (1, num_anchors, 4)
builder.add_permute(name="permute_output",
                    dim=(0, 3, 2, 1),
                    input_name="concat_output",
                    output_name="raw_coordinates")

decoder_model = coremltools.models.MLModel(builder.spec)
decoder_model.save("Decoder.mlmodel")


# ===============================
# PART 3: Non-maximum suppression
# ===============================

nms_spec = coremltools.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3

for i in range(2):
    decoder_output = decoder_model._spec.description.output[i].SerializeToString()

    nms_spec.description.input.add()
    nms_spec.description.input[i].ParseFromString(decoder_output)

    nms_spec.description.output.add()
    nms_spec.description.output[i].ParseFromString(decoder_output)
    
nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

output_sizes = [num_classes, 4]
for i in range(2):
    ma_type = nms_spec.description.output[i].type.multiArrayType
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = -1
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
    ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
    del ma_type.shape[:]

nms = nms_spec.nonMaximumSuppression
nms.confidenceInputFeatureName = "raw_confidence"
nms.coordinatesInputFeatureName = "raw_coordinates"
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThresholdInputFeatureName = "iouThreshold"
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

default_iou_threshold = 0.6
default_confidence_threshold = 0.4
nms.iouThreshold = default_iou_threshold
nms.confidenceThreshold = default_confidence_threshold

nms.pickTop.perClass = True

labels = np.loadtxt("coco_labels.txt", dtype=str, delimiter="\n")
nms.stringClassLabels.vector.extend(labels)

nms_model = coremltools.models.MLModel(nms_spec)
nms_model.save("NMS.mlmodel")


# ===============================================
# PART 4: Putting it all together into a pipeline
# ===============================================

from coremltools.models.pipeline import *

input_features = [ ("image", datatypes.Array(3, 300, 300)),
                   ("iouThreshold", datatypes.Double()),
                   ("confidenceThreshold", datatypes.Double()) ]

output_features = [ "confidence", "coordinates" ]

pipeline = Pipeline(input_features, output_features)

# We added a dimension of size 1 to the back of the inputs of the decoder 
# model, so we should also add this to the output of the SSD model or else 
# the inputs and outputs do not match and the pipeline is not valid.
ssd_output = ssd_model._spec.description.output
ssd_output[0].type.multiArrayType.shape[:] = [num_classes + 1, num_anchors, 1]
ssd_output[1].type.multiArrayType.shape[:] = [4, num_anchors, 1]

pipeline.add_model(ssd_model)
pipeline.add_model(decoder_model)
pipeline.add_model(nms_model)

# The "image" input should really be an image, not a multi-array.
pipeline.spec.description.input[0].ParseFromString(ssd_model._spec.description.input[0].SerializeToString())

# Copy the declarations of the "confidence" and "coordinates" outputs.
# The Pipeline makes these strings by default.
pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

# Add descriptions to the inputs and outputs.
pipeline.spec.description.input[1].shortDescription = "(optional) IOU Threshold override"
pipeline.spec.description.input[2].shortDescription = "(optional) Confidence Threshold override"
pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 Class confidence"
pipeline.spec.description.output[1].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"

# Add metadata to the model.
pipeline.spec.description.metadata.versionString = "ssdlite_mobilenet_v2_coco_2018_05_09"
pipeline.spec.description.metadata.shortDescription = "MobileNetV2 + SSDLite, trained on COCO"
pipeline.spec.description.metadata.author = "Converted to Core ML by Matthijs Hollemans"
pipeline.spec.description.metadata.license = "https://github.com/tensorflow/models/blob/master/research/object_detection"

# Add the list of class labels and the default threshold values too.
user_defined_metadata = {
    "iou_threshold": str(default_iou_threshold),
    "confidence_threshold": str(default_confidence_threshold),
    "classes": ",".join(labels)
}
pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)

# Don't forget this or Core ML might attempt to run the model on an unsupported
# operating system version!
pipeline.spec.specificationVersion = 3

final_model = coremltools.models.MLModel(pipeline.spec)
final_model.save(coreml_model_path)

print(final_model)
print("Done!")
