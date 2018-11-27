def get_nn(spec):
    if spec.WhichOneof("Type") == "neuralNetwork":
        return spec.neuralNetwork
    elif spec.WhichOneof("Type") == "neuralNetworkClassifier":
        return spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
        return spec.neuralNetworkRegressor
    else:
        raise ValueError("MLModel does not have a neural network")


def convert_multiarray_to_image(feature, is_bgr=False):
    import coremltools.proto.FeatureTypes_pb2 as ft

    if feature.type.WhichOneof("Type") != "multiArrayType":
        raise ValueError("%s is not a multiarray type" % feature.name)

    shape = tuple(feature.type.multiArrayType.shape)
    channels = None
    if len(shape) == 2:
        channels = 1
        height, width = shape
    elif len(shape) == 3:
        channels, height, width = shape

    if channels != 1 and channels != 3:
        raise ValueError("Shape {} not supported for image type".format(shape))

    if channels == 1:
        feature.type.imageType.colorSpace = ft.ImageFeatureType.GRAYSCALE
    elif channels == 3:
        if is_bgr:
            feature.type.imageType.colorSpace = ft.ImageFeatureType.BGR
        else:
            feature.type.imageType.colorSpace = ft.ImageFeatureType.RGB

    feature.type.imageType.width = width
    feature.type.imageType.height = height
