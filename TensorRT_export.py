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
# 返回文件/目录的大小（MB）
def file_size(path):
    # Return file/dir size (MB) 
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path =  Path(path)
    if path.is_file():
        return path.stat().st_size / mb # 返回文件大小
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb # 返回目录中所有文件的大小之和
    else:
        return 0.0  # 路径既不是文件也不是目录，返回0.0

# 获取函数func()的默认参数
def get_default_args(func):
    # Get func() default arguments
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

# Profile类。用法：@Profile()装饰器或'with Profile():'上下文管理器
class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t # 累计时间
        self.cuda = torch.cuda.is_available() # 检查是否可用CUDA

    def __enter__(self):
        self.start = self.time() # 记录开始时间
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # 计算时间差 delta-time
        self.t += self.dt  # 累加时间差

    def time(self):
        if self.cuda:
            torch.cuda.synchronize() # 等待CUDA操作完成
        return time.time() # 返回当前时间戳

# 导出装饰器，即 @try_export
def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs) # 执行内部函数
            print(f'export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)') # 打印导出成功的消息
            return f, model # 返回导出的结果
        except Exception as e:
            print(f'export failure ❌ {dt.t:.1f}s: {e}') # 打印导出失败的消息
            return None, None # 返回空结果

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
