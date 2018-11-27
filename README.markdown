# Core ML Survival Guide

This is the source code for the book [Core ML Survival Guide](http://leanpub.com/coreml-survival-guide).

## The book

If you don't have the book yet, here's my sales pitch: The [Core ML Survival Guide](http://leanpub.com/coreml-survival-guide) is more than 60 chapters and 300+ pages of Core ML tips and tricks. 

Learn the best ways to convert your models to Core ML, how the mlmodel file format works, how to perform model surgery to fix issues with your mlmodel files, how use `MLMultiArray`, and much more!

If you're serious about Core ML, [you need this book](http://leanpub.com/coreml-survival-guide). Seriously! :smile:

## The source code

Included in this source code repo is the following:

- **CheckInputImage:** Demo app for iOS that shows how to use a very basic image-to-image model to verify how Vision and the neural network preprocessing options modify the input image.

- Scripts/
    - **convert_multiarrays_to_floats.py:** Changes the data type of any MultiArray inputs and outputs from `DOUBLE` to `FLOAT32`.
    - **convert_to_float16.py:** Changes the weights of the model from 32-bit to 16-bit floating point, cutting the model size in half.
    - **deeplab.py:** Demonstration of how to clean up a model by renaming inputs and outputs, removing unnecessary layers, inserting new layers, etc.
    - **helpers.py:** Useful utility functions for model surgery.
    - **quantize.py:** For quantizing and dequantizing models.

Also check out my other repo [CoreMLHelpers](https://github.com/hollance/CoreMLHelpers) for a collection of helper code that makes it a little easier to work with Core ML in Swift.
