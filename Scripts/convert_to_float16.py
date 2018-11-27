import sys, coremltools  

if len(sys.argv) != 3:  
    print("USAGE: %s <input_mlmodel> <output_mlmodel>" % sys.argv[0])
    sys.exit(1)  

input_model_path = sys.argv[1]  
output_model_path = sys.argv[2]  

spec = coremltools.utils.load_spec(input_model_path)  
spec_fp16 = coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)
coremltools.utils.save_spec(spec_fp16, output_model_path)
