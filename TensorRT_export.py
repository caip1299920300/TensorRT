import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch,time
import contextlib
import inspect
from pathlib import Path

# 加载ONNX模型
#model = onnx.load(onnx_model_path)
#onnx.checker.check_model(model)  # check onnx model
#import onnxsim
#model, check = onnxsim.simplify(model)
#assert check, 'assert check failed'
#onnx.save(model, onnx_model_path)


import torch,time
import contextlib
import inspect
from pathlib import Path
def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path =  Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0

def get_default_args(func):
    # Get func() default arguments
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()

def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            print(f'export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            print(f'export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func

@try_export
def export_engine(onnx_model_path,engine_path, workspace=4):
    # 创建TensorRT的Logger对象
    logger = trt.Logger(trt.Logger.INFO)
    # 创建TensorRT的Builder对象
    builder = trt.Builder(logger)
    # 指定TensorRT的优化器配置
    config = builder.create_builder_config()
    config.max_workspace_size = workspace*1 << 30  # 设置最大工作空间大小

    # 解析ONNX模型并构建TensorRT网络
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    # tensorrt读取onnx模型，初始化
    #if not parser.parse(model.SerializeToString()):
    if not parser.parse_from_file(str(onnx_model_path)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_model_path}')
    # 构建并优化TensorRT引擎
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError('Failed to build TensorRT engine')
    # 保存TensorRT引擎
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    return engine_path, None

if __name__ == "__main__":
    onnx_model_path = './best.onnx'
    engine_path = './best.engine'
    export_engine(onnx_model_path,engine_path)
