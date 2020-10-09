import sys
import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft

def update_multiarray_to_float32(feature):
    if feature.type.HasField("multiArrayType"):
        feature.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32

if len(sys.argv) != 3:
    print("USAGE: %s <input_mlmodel> <output_mlmodel>" % sys.argv[0])
    sys.exit(1)

input_model_path = sys.argv[1]
output_model_path = sys.argv[2]

spec = ct.utils.load_spec(input_model_path)

for feature in spec.description.input:
    update_multiarray_to_float32(feature)

for feature in spec.description.output:
    update_multiarray_to_float32(feature)

ct.utils.save_spec(spec, output_model_path)
