import coremltools
import coremltools.proto.FeatureTypes_pb2 as ft

spec = coremltools.proto.Model_pb2.Model()
spec.specificationVersion = 1

new_input = spec.description.input.add()
new_input.name = "image"
new_input.type.imageType.width = 256
new_input.type.imageType.height = 256
new_input.type.imageType.colorSpace = ft.ImageFeatureType.RGB

new_output = spec.description.output.add()
new_output.name = "generatedImage"
new_output.type.imageType.width = 256
new_output.type.imageType.height = 256
new_output.type.imageType.colorSpace = ft.ImageFeatureType.RGB

# If you want the output to be a MultiArray instead of an image:
#new_output.type.multiArrayType.shape.extend([3, 256, 256])
#new_output.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32

new_prepro = spec.neuralNetwork.preprocessing.add()
new_prepro.scaler.channelScale = 1.0
new_prepro.scaler.redBias = 0.0
new_prepro.scaler.greenBias = 0.0
new_prepro.scaler.blueBias = 0.0
new_prepro.featureName = spec.description.input[0].name

new_layer = spec.neuralNetwork.layers.add()
new_layer.name = "test_layer"
new_layer.input.append(spec.description.input[0].name)
new_layer.output.append(spec.description.output[0].name)
new_layer.activation.linear.alpha = 1.0

print(spec.description)

coremltools.utils.save_spec(spec, "Image2Image.mlmodel")
