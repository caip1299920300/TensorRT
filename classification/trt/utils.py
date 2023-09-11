
# import pdb;pdb.set_trace()
import sys
sys.path.append('trt')
import common
import tensorrt as trt
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

# The Onnx path is used for Onnx models.
def build_engine_onnx(TRT_LOGGER, model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # builder.max_workspace_size = common.GiB(1)
        config = builder.create_builder_config()
        config.max_workspace_size = common.GiB(1)
        # import pdb;pdb.set_trace()
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
        return engine
        # return builder.build_cuda_engine(network)

def build_engine_onnx_int8(TRT_LOGGER, model_file, calib):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        config = builder.create_builder_config()
        config.max_workspace_size = common.GiB(1)
        # import pdb;pdb.set_trace()
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator  = calib
        # import pdb;pdb.set_trace()
        profile = builder.create_optimization_profile()
         # 设置网络的输入：set_shape()方法需要传递三个参数，分别是最小形状(min)，优化形状(opt)，和最大形状(max)。
        profile.set_shape("input", (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
        return engine
        # return builder.build_cuda_engine(network)

def build_engine(TRT_LOGGER, model_file):
    with open(model_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
